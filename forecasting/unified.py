"""Evaluación común de regresores directos y referencias temporales."""
from __future__ import annotations

import hashlib
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import Holt
from threadpoolctl import threadpool_limits

from .direct import ModelSpec, build_design, fit_direct
from .experiment import ROOT, SEED, digest_json

BASELINE_LABELS = {
    'persistence': 'Persistencia', 'annual_trend': 'Tendencia interanual',
    'damped_trend': 'Tendencia reciente amortiguada', 'holt': 'Holt log amortiguado',
    'recursive_huber': 'Huber crecimiento recursivo',
}


def baseline_specs() -> list[ModelSpec]:
    return [ModelSpec(name, 'ar', {}) for name in BASELINE_LABELS]


def read_panel(path: Path) -> pd.DataFrame:
    panel = pd.read_parquet(path)
    panel['quarter'] = pd.PeriodIndex(panel['quarter'], freq='Q')
    panel['province_code'] = panel['province_code'].astype(str).str.zfill(2)
    if panel.duplicated(['province_code', 'quarter']).any():
        raise ValueError('Claves provinciales duplicadas.')
    return panel.sort_values(['province_code', 'quarter']).reset_index(drop=True)


def frame_digest(frame: pd.DataFrame) -> str:
    data = frame.copy().sort_values(['province_code', 'quarter']).reset_index(drop=True)
    for col in data:
        if isinstance(data[col].dtype, pd.PeriodDtype):
            data[col] = data[col].astype(str)
    return hashlib.sha256(pd.util.hash_pandas_object(data, index=False).values.tobytes()).hexdigest()


def code_digest() -> str:
    files = ['forecasting/direct.py', 'forecasting/unified.py', 'forecasting/experiment.py',
             'forecasting/candidate_models.py', 'forecasting/calendar.py']
    return hashlib.sha256(b''.join((ROOT / f).read_bytes() for f in files)).hexdigest()


def _local_prediction(panel: pd.DataFrame, origin: pd.Period, horizon: int,
                      family: str, rows: pd.DataFrame) -> tuple[np.ndarray, int]:
    values, fallbacks = [], 0
    for code in rows['province_code']:
        group = panel.loc[panel.province_code.eq(code) & panel.quarter.le(origin)]
        series = group.set_index('quarter')['valor_tasado_m2'].astype(float)
        series = series.reindex(pd.period_range(series.index.min(), origin, freq='Q'))
        last = float(series.iloc[-1])
        if not np.isfinite(last) or last <= 0:
            raise ValueError(f'Precio del origen inválido: {code}, {origin}')
        if family == 'persistence':
            prediction = last
        elif family == 'annual_trend':
            previous = series.get(origin - 4, np.nan)
            if pd.isna(previous):
                prediction, fallbacks = last, fallbacks + 1
            else:
                prediction = last * (last / previous) ** (horizon / 4)
        elif family == 'damped_trend':
            changes = np.log(series).diff().iloc[-8:].dropna()
            slope = float(changes.median()) if len(changes) else 0.0
            prediction = last * np.exp(slope * (1 - .85 ** horizon) / (1 - .85))
        elif family == 'holt':
            # No comprimir el tiempo al eliminar ausencias: usar el último tramo continuo.
            missing = np.flatnonzero(series.isna().to_numpy())
            continuous = series.iloc[missing[-1] + 1:] if len(missing) else series
            if len(continuous) < 8:
                prediction, fallbacks = last, fallbacks + 1
            else:
                fitted = Holt(np.log(continuous.to_numpy()), damped_trend=True,
                              initialization_method='estimated').fit(optimized=True)
                prediction = float(np.exp(fitted.forecast(horizon)[-1]))
        else:
            raise ValueError(family)
        values.append(prediction)
    result = np.array(values)
    if not np.isfinite(result).all() or not (result > 0).all():
        raise ValueError('Predicciones no positivas/no finitas.')
    return result, fallbacks


def predict_at_origin(panel: pd.DataFrame, spec: ModelSpec, horizon: int,
                      origin: object, design: pd.DataFrame | None = None,
                      require_actual: bool = True) -> tuple[pd.DataFrame, dict]:
    origin = pd.Period(origin, freq='Q')
    design = build_design(panel, horizon) if design is None else design
    rows = design.loc[design.quarter.eq(origin) & design.valor_tasado_m2.gt(0)].copy()
    if require_actual:
        rows = rows.loc[rows.target_price.gt(0)]
    if rows.empty:
        raise ValueError(f'Sin observaciones evaluables en {origin}, h={horizon}')
    metadata = {'origin': str(origin), 'horizon': horizon, 'candidate_id': spec.key,
                'family': spec.family, 'feature_set': spec.feature_set, 'warnings': []}
    with warnings.catch_warnings(record=True) as caught, threadpool_limits(limits=1):
        warnings.simplefilter('always')
        if spec.family == 'recursive_huber':
            from .candidate_models import GlobalHuberGrowthModel
            # Cortar también físicamente el panel; no exponer el futuro a fit.
            observed = panel.loc[panel.quarter.le(origin) & panel.province_code.isin(rows.province_code)]
            fitted = GlobalHuberGrowthModel().fit(observed, origin)
            forecast = fitted.forecast(horizon)
            mapped = forecast.loc[forecast.horizon.eq(horizon)].set_index('province_code').prediction
            prediction = rows.province_code.map(mapped).to_numpy()
        elif spec.family in BASELINE_LABELS:
            prediction, fallbacks = _local_prediction(panel, origin, horizon, spec.family, rows)
            metadata['fallbacks'] = fallbacks
        else:
            fitted = fit_direct(spec, design, origin)
            prediction = fitted.predict(rows)
            metadata.update(n_train=fitted.n_train, training_target_max=str(fitted.training_target_max),
                            numeric_features=fitted.numeric_features,
                            excluded_features=fitted.excluded_features,
                            warnings=fitted.fit_warnings)
        metadata['warnings'] += [f'{type(w.message).__name__}: {w.message}' for w in caught]
    if not np.isfinite(prediction).all() or not (prediction > 0).all():
        raise ValueError('Predicción inválida; candidato no elegible.')
    result = pd.DataFrame({
        'candidate_id': spec.key, 'model': BASELINE_LABELS.get(spec.family, spec.family),
        'family': spec.family, 'feature_set': spec.feature_set,
        'province_code': rows.province_code.to_numpy(),
        'province': rows['province'].to_numpy() if 'province' in rows else rows.province_code.to_numpy(),
        'origin': str(origin), 'forecast_quarter': str(origin + horizon), 'horizon': horizon,
        'prediction': prediction, 'actual': rows.target_price.to_numpy(),
        'current_price': rows.valor_tasado_m2.to_numpy(),
    })
    result['error'] = result.prediction - result.actual
    result['absolute_error'] = result.error.abs()
    result['squared_error'] = result.error ** 2
    result['ape_pct'] = 100 * result.absolute_error / result.actual
    return result, metadata


def evaluate_task(panel: pd.DataFrame, spec_data: dict, horizon: int, origins: list[str],
                  target_cutoff: str, cache_root: str, fingerprint: str) -> tuple[pd.DataFrame, list[dict]]:
    spec = ModelSpec.from_dict(spec_data)
    cache = Path(cache_root) / digest_json([spec_data, horizon, origins, target_cutoff, fingerprint])
    frame_path, meta_path = cache.with_suffix('.parquet'), cache.with_suffix('.json')
    if frame_path.exists() and meta_path.exists():
        return pd.read_parquet(frame_path), json.loads(meta_path.read_text(encoding='utf-8'))
    cutoff = pd.Period(target_cutoff, freq='Q')
    # Construir transformaciones únicamente con el tramo autorizado de cada fase.
    data = panel.loc[panel.quarter.le(cutoff)].copy()
    design = build_design(data, horizon)
    outputs, metadata = [], []
    for text in origins:
        origin = pd.Period(text, freq='Q')
        if origin + horizon > cutoff:
            continue
        try:
            result, record = predict_at_origin(data, spec, horizon, origin, design)
            outputs.append(result)
            metadata.append(record)
        except Exception as exc:
            metadata.append({'candidate_id': spec.key, 'family': spec.family, 'horizon': horizon,
                             'origin': str(origin), 'failure': f'{type(exc).__name__}: {exc}'})
    output = pd.concat(outputs, ignore_index=True) if outputs else pd.DataFrame()
    cache.parent.mkdir(parents=True, exist_ok=True)
    output.to_parquet(frame_path, index=False)
    meta_path.write_text(json.dumps(metadata, ensure_ascii=False, default=str), encoding='utf-8')
    return output, metadata


def metric_table(detail: pd.DataFrame, groups: list[str]) -> pd.DataFrame:
    rows = []
    for key, part in detail.groupby(groups, sort=True, dropna=False):
        keys = key if isinstance(key, tuple) else (key,)
        row = dict(zip(groups, keys))
        row.update(observations=len(part), origins=part.origin.nunique(),
                   MAE=float(part.absolute_error.mean()),
                   RMSE=float(np.sqrt(part.squared_error.mean())),
                   MAPE_pct=float(part.ape_pct.mean()), bias=float(part.error.mean()))
        # Escala aleatoria local: error relativo a la variación típica de cuatro trimestres
        # se informa aparte de MAE; no se interpreta el R2 territorial como habilidad temporal.
        rows.append(row)
    return pd.DataFrame(rows)


def pairwise_origin_bootstrap(detail: pd.DataFrame, references: list[str], replicas: int = 2000) -> pd.DataFrame:
    """IC descriptivo por bloques móviles de cuatro orígenes, provincias juntas."""
    rows, rng = [], np.random.default_rng(SEED)
    selected = detail.loc[detail.role.eq('selected')]
    for reference in references:
        baseline = detail.loc[detail.role.eq(reference)]
        pair = selected.merge(baseline, on=['province_code', 'origin', 'horizon'], suffixes=('_sel', '_ref'))
        pair['gain'] = pair.absolute_error_ref - pair.absolute_error_sel
        for horizon, part in pair.groupby('horizon'):
            scores = part.groupby('origin', sort=True).gain.mean().to_numpy()
            n = len(scores)
            if n < 8:
                low = high = np.nan
            else:
                block = min(4, n)
                means = []
                for _ in range(replicas):
                    starts = rng.integers(0, n - block + 1, size=int(np.ceil(n / block)))
                    sample = np.concatenate([scores[s:s + block] for s in starts])[:n]
                    means.append(sample.mean())
                low, high = np.quantile(means, [.025, .975])
            rows.append({'reference': reference, 'horizon': horizon, 'origins': n,
                         'MAE_improvement': scores.mean(), 'lower95_descriptive': low,
                         'upper95_descriptive': high,
                         'note': 'insufficient_origins' if n < 8 else 'moving_origin_blocks_4'})
    return pd.DataFrame(rows)
