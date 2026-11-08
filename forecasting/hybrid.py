"""Combinaciones convexas exploratorias; los pesos solo se ajustan en desarrollo."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .experiment import HORIZONS, SELECTION_CUTOFF, TUNING_TARGETS

KEYS = ['province_code', 'origin', 'horizon']
BASES = ('annual_trend', 'holt')
WEIGHTS = tuple(i / 10 for i in range(11))


def aligned_predictions(detail: pd.DataFrame) -> pd.DataFrame:
    """Exige muestras idénticas, fechas coherentes y precios válidos por componente."""
    roles = ['selected', 'persistence', *BASES]
    required = {*KEYS, 'role', 'prediction', 'actual', 'forecast_quarter', 'current_price'}
    if not required.issubset(detail):
        raise ValueError('Faltan columnas de predicción.')
    relevant = detail.loc[detail.role.isin(roles)].copy()
    if relevant.duplicated(['role', *KEYS]).any():
        raise ValueError('Predicciones duplicadas.')
    parts = {}
    for role in roles:
        part = relevant.loc[relevant.role.eq(role)].set_index(KEYS).sort_index()
        if part.empty or set(part.index.get_level_values('horizon')) != set(HORIZONS):
            raise ValueError('Faltan componentes u horizontes.')
        values = part[['prediction', 'actual', 'current_price']].to_numpy(dtype=float)
        if not (np.isfinite(values) & (values > 0)).all():
            raise ValueError('Precios no positivos o no finitos.')
        origins = pd.PeriodIndex(part.index.get_level_values('origin'), freq='Q')
        targets = pd.PeriodIndex(part.forecast_quarter, freq='Q')
        horizons = part.index.get_level_values('horizon').to_numpy(dtype=int)
        if not np.array_equal(origins.asi8 + horizons, targets.asi8):
            raise ValueError('Calendario incoherente.')
        parts[role] = part
    selected = parts['selected']
    result = selected[['forecast_quarter', 'actual', 'current_price']].copy()
    for role, part in parts.items():
        if not part.index.equals(selected.index):
            raise ValueError('Muestras distintas entre componentes.')
        for column in ('actual', 'current_price', 'forecast_quarter'):
            if not part[column].equals(selected[column]):
                raise ValueError(f'{column} distinto entre componentes.')
        result[role] = part.prediction
    return result.reset_index()


def blend(base, selected, weight: float) -> np.ndarray:
    """Peso de ML en escala de precios; 0=referencia pura, 1=selección original."""
    if not np.isfinite(weight) or not 0 <= weight <= 1:
        raise ValueError('El peso debe estar entre cero y uno.')
    return (1 - weight) * np.asarray(base) + weight * np.asarray(selected)


def fit_weights(development: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Un peso global por base, media de MAE relativo por horizonte, empate a menor ML.

    Incluye extremos para no forzar una combinación si el desarrollo no la respalda.
    No admite predicciones de calibración ni evaluación.
    """
    pairs = aligned_predictions(development)
    dates = pd.PeriodIndex(pairs.forecast_quarter, freq='Q')
    if (dates > pd.Period(SELECTION_CUTOFF)).any():
        raise ValueError('El ajuste no admite objetivos posteriores a desarrollo.')
    for _, part in pairs.groupby('horizon'):
        if set(part.forecast_quarter) != set(TUNING_TARGETS):
            raise ValueError('Fechas de selección distintas del protocolo.')
    reference = (pairs.persistence - pairs.actual).abs().groupby(pairs.horizon).mean()
    if reference.le(0).any():
        raise ValueError('MAE de persistencia nulo; escala relativa indefinida.')
    rows = []
    for base in BASES:
        for weight in WEIGHTS:
            errors = pd.Series(np.abs(blend(pairs[base], pairs.selected, weight) - pairs.actual),
                               index=pairs.index)
            mae = errors.groupby(pairs.horizon).mean()
            rows.append({'base': base, 'ml_weight': weight,
                         'mean_relative_MAE': float((mae / reference).mean()),
                         'mean_horizon_MAE': float(mae.mean())})
    grid = pd.DataFrame(rows).sort_values(['mean_relative_MAE', 'ml_weight', 'base'])
    best = grid.drop_duplicates('base')
    frozen = {'weights': dict(zip(best.base, best.ml_weight)),
              'preferred_base': str(best.iloc[0].base),
              'preferred_ml_weight': float(best.iloc[0].ml_weight)}
    return grid.reset_index(drop=True), frozen


def apply_weights(detail: pd.DataFrame, frozen: dict) -> pd.DataFrame:
    """Aplica pesos ya fijados, sin consultar errores para modificarlos."""
    pairs = aligned_predictions(detail)
    outputs = []
    if set(frozen['weights']) != set(BASES):
        raise ValueError('Pesos incompletos.')
    for base in BASES:
        weight = float(frozen['weights'][base])
        result = pairs[KEYS + ['forecast_quarter', 'actual', 'current_price']].copy()
        result['role'] = f'hybrid_{base}'
        result['base'] = base
        result['ml_weight'] = weight
        result['prediction'] = blend(pairs[base], pairs.selected, weight)
        result['error'] = result.prediction - result.actual
        result['absolute_error'] = result.error.abs()
        result['squared_error'] = result.error ** 2
        result['ape_pct'] = 100 * result.absolute_error / result.actual
        outputs.append(result)
    return pd.concat(outputs, ignore_index=True)
