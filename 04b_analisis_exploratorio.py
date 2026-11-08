"""EDA reproducible: asociaciones de desarrollo y diagnóstico descriptivo separado.

Las correlaciones y la información mutua usan únicamente orígenes Y objetivos
observados hasta 2018Q4. No seleccionan automáticamente variables ni interpretan
asociaciones como efectos causales. Los datos posteriores se usan exclusivamente
en informes de cobertura y evolución, identificados como descriptivos.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_regression

from forecasting.direct import AR_FEATURES, EXTERNAL_BLOCKS, FEATURE_BLOCKS, build_design


ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT = ROOT / "data/processed/panel_provincial_trimestral.parquet"
DEFAULT_OUTPUT = ROOT / "data/reports/eda"
DEVELOPMENT_END = "2018Q4"
INITIAL_TRAIN_END = "2010Q4"
HORIZONS = (1, 4, 8, 12)
PRICE = "valor_tasado_m2"
MIN_PAIRS = 40
REGIMES = (("1995-2007", "1995Q1", "2007Q4"),
           ("2008-2013", "2008Q1", "2013Q4"),
           ("2014-2018", "2014Q1", "2018Q4"))


def normalize_panel(panel: pd.DataFrame) -> pd.DataFrame:
    """Valida claves y tipos; no completa huecos ni rellena valores."""
    required = {"province_code", "quarter", PRICE}
    missing = required - set(panel)
    if missing:
        raise ValueError(f"Faltan columnas: {sorted(missing)}")
    out = panel.copy()
    if out[["province_code", "quarter"]].isna().any().any():
        raise ValueError("Las claves provinciales y trimestrales deben estar completas.")
    out["province_code"] = out["province_code"].astype(str).str.zfill(2)
    out["quarter"] = pd.PeriodIndex(out["quarter"], freq="Q")
    if out.duplicated(["province_code", "quarter"]).any():
        raise ValueError("Existen claves provincia-trimestre duplicadas.")
    return out.sort_values(["province_code", "quarter"]).reset_index(drop=True)


def economic_block(feature: str) -> str:
    """Taxonomía exhaustiva de EDA, sin presuponer utilidad predictiva."""
    if feature in AR_FEATURES or feature.startswith(("valor_tasado", "growth_", "log_price", "yoy_")):
        return "historia_precio"
    if "hipoteca" in feature or "euribor" in feature:
        return "credito"
    if "hotel" in feature:
        return "turismo"
    if "poblacion" in feature or "migratorio" in feature:
        return "demografia"
    if feature.startswith(("trans_", "compraventas")):
        return "actividad"
    if feature.startswith(("iniciadas_", "terminadas_", "vp_")) or "suelo" in feature:
        return "oferta"
    if feature in {"superficie_km2", "longitud_costa_km"}:
        return "territorio"
    if any(token in feature for token in ("renta", "pib", "empleo", "paro", "ipc", "vab_")):
        return "economia"
    return "otros"


def model_blocks(feature: str) -> str:
    return ";".join(name for name, columns in FEATURE_BLOCKS.items() if feature in columns)


def add_model_features(panel: pd.DataFrame) -> pd.DataFrame:
    """Expone exactamente las transformaciones AR del benchmark definitivo."""
    out = normalize_panel(panel)
    design = build_design(out, horizon=1)
    for feature in AR_FEATURES:
        out[feature] = design[feature].to_numpy()
    return out


def numeric_predictors(panel: pd.DataFrame) -> list[str]:
    excluded = {"province_code", "year", "quarter_number", "split", "price_observed"}
    return sorted(column for column in panel.select_dtypes(include=np.number)
                  if column not in excluded
                  and not column.startswith(("target_", "future_", "label_", "audit_"))
                  and not column.endswith(("_available", "_observed", "_imputed")))


def exact_labels(panel: pd.DataFrame, horizon: int,
                 cutoff: str | pd.Period = DEVELOPMENT_END) -> pd.DataFrame:
    """100*log(P[t+h]/P[t]), buscando t+h por clave, nunca por fila.

    Se corta el calendario antes del lookup: ni siquiera etiquetas presentes en
    el fichero de modelado pueden filtrar un valor posterior al corte.
    """
    if horizon < 1:
        raise ValueError("El horizonte debe ser positivo.")
    data = normalize_panel(panel)
    end = pd.Period(cutoff, freq="Q")
    data = data.loc[data["quarter"] <= end].copy()
    data["target_quarter"] = data["quarter"] + horizon
    lookup = data.set_index(["province_code", "quarter"])[PRICE]
    keys = pd.MultiIndex.from_arrays([data["province_code"], data["target_quarter"]])
    future = lookup.reindex(keys).to_numpy(dtype=float, na_value=np.nan)
    current = data[PRICE].to_numpy(dtype=float, na_value=np.nan)
    valid = (current > 0) & (future > 0) & (data["target_quarter"] <= end).to_numpy()
    growth = np.full(len(data), np.nan)
    growth[valid] = 100 * np.log(future[valid] / current[valid])
    data["forward_log_growth"] = growth
    return data


def exact_feature_lags(data: pd.DataFrame, features: Iterable[str], lag: int) -> pd.DataFrame:
    """Retardo adicional sobre una variable ya ajustada por disponibilidad."""
    if lag < 0:
        raise ValueError("No se permiten adelantos de predictores.")
    out = normalize_panel(data)
    keys = pd.MultiIndex.from_arrays([out["province_code"], out["quarter"] - lag])
    lookup = out.set_index(["province_code", "quarter"])
    for feature in features:
        out[feature] = lookup[feature].reindex(keys).to_numpy(dtype=float, na_value=np.nan)
    return out


def _correlation_rows(data: pd.DataFrame, features: Iterable[str], target: str,
                      context: dict) -> list[dict]:
    rows = []
    valid_target = data[target].notna()
    target_count = int(valid_target.sum())
    for feature in features:
        pairs = data.loc[valid_target & data[feature].notna(),
                         ["province_code", feature, target]].copy()
        # A contemporaneous price correlation with itself is not informative.
        if feature == target:
            continue
        pairs = pairs.replace([np.inf, -np.inf], np.nan).dropna()
        count = len(pairs)
        pooled = within = np.nan
        if count >= MIN_PAIRS and pairs[feature].nunique() > 1 and pairs[target].nunique() > 1:
            pooled = pairs[feature].corr(pairs[target], method="spearman")
            numeric = pairs[[feature, target]].astype(float)
            demeaned = numeric - numeric.groupby(pairs["province_code"]).transform("mean")
            if demeaned[feature].std() > 1e-12 and demeaned[target].std() > 1e-12:
                within = demeaned[feature].corr(demeaned[target], method="spearman")
        rows.append({**context, "variable": feature, "bloque": economic_block(feature),
                     "bloques_modelado": model_blocks(feature),
                     "n_pares": count, "n_provincias": pairs["province_code"].nunique(),
                     "cobertura_etiquetados_pct": 100 * count / target_count if target_count else np.nan,
                     "spearman_agrupada": pooled, "spearman_sin_media_provincial": within})
    return rows


def selection_reports(panel: pd.DataFrame, cutoff: str = DEVELOPMENT_END,
                      horizons: Iterable[int] = HORIZONS) -> dict[str, pd.DataFrame]:
    """Informes de asociación: todas las operaciones se ajustan en desarrollo.

    MI es un diagnóstico univariante in-sample (no una puntuación de validación).
    Cada mediana se estima solo con filas de desarrollo cuyo objetivo existe.
    """
    data = normalize_panel(panel)
    data = data.loc[data["quarter"] <= pd.Period(cutoff, freq="Q")].copy()
    data = add_model_features(data)
    features = numeric_predictors(data)
    correlations, stability, mi_rows, lag_rows = [], [], [], []
    external = sorted({column for block in EXTERNAL_BLOCKS.values() for column in block
                       if column in features})
    levels = _correlation_rows(data, features, PRICE, {"objetivo": "precio_contemporaneo"})
    for horizon in horizons:
        labeled = exact_labels(data, horizon, cutoff)
        correlations.extend(_correlation_rows(
            labeled, features, "forward_log_growth", {"horizonte": horizon}))
        for lag in (0, 1, 4):
            delayed = exact_feature_lags(labeled, external, lag)
            lag_rows.extend(_correlation_rows(delayed, external, "forward_log_growth",
                            {"horizonte": horizon, "retardo_adicional": lag}))
        for name, first, last in REGIMES:
            window = labeled.loc[labeled["quarter"].between(pd.Period(first), pd.Period(last))
                                 & (labeled["target_quarter"] <= pd.Period(last))]
            stability.extend(_correlation_rows(
                window, features, "forward_log_growth", {"horizonte": horizon, "regimen": name}))
        train = labeled.loc[labeled["forward_log_growth"].notna()].copy()
        for feature in features:
            series = train[feature].astype(float).replace([np.inf, -np.inf], np.nan)
            available = series.notna()
            score = median = np.nan
            if available.sum() >= MIN_PAIRS and series.nunique() > 1 and len(train) > 3:
                median = float(series.median())
                filled = series.fillna(median).to_numpy().reshape(-1, 1)
                score = float(mutual_info_regression(
                    filled, train["forward_log_growth"].to_numpy(),
                    random_state=42, n_neighbors=5)[0])
            mi_rows.append({"horizonte": horizon, "variable": feature,
                            "bloque": economic_block(feature), "bloques_modelado": model_blocks(feature),
                            "informacion_mutua_nats": score,
                            "n_objetivos": len(train), "n_observados": int(available.sum()),
                            "n_imputados": int((~available).sum()), "mediana_desarrollo": median})
    matrix = data[features].astype(float).corr(method="spearman", min_periods=MIN_PAIRS)
    redundancy = []
    for i, first in enumerate(features):
        for second in features[i + 1:]:
            rho = matrix.loc[first, second]
            if pd.notna(rho) and abs(rho) >= .85:
                redundancy.append({"variable_a": first, "variable_b": second,
                                   "spearman": rho,
                                   "n_pares": int(data[[first, second]].notna().all(axis=1).sum())})
    return {"correlaciones_nivel": pd.DataFrame(levels),
            "correlaciones_crecimiento": pd.DataFrame(correlations),
            "correlaciones_estabilidad": pd.DataFrame(stability),
            "correlaciones_retardos_externos": pd.DataFrame(lag_rows),
            "informacion_mutua": pd.DataFrame(mi_rows),
            "redundancia": pd.DataFrame(redundancy, columns=["variable_a", "variable_b", "spearman", "n_pares"])}


def coverage_reports(panel: pd.DataFrame, cutoff: str = DEVELOPMENT_END,
                     initial_train_end: str = INITIAL_TRAIN_END) -> dict[str, pd.DataFrame]:
    """Cobertura completa descriptiva, con columnas de desarrollo diferenciadas."""
    data = add_model_features(panel)
    features = numeric_predictors(data)
    dev = data["quarter"] <= pd.Period(cutoff)
    train = data["quarter"] <= pd.Period(initial_train_end)
    coverage, by_year, quantiles, calendars = [], [], [], []
    for feature in features:
        values = data[feature].astype(float).replace([np.inf, -np.inf], np.nan)
        dates = data.loc[values.notna(), "quarter"]
        coverage.append({"variable": feature, "bloque": economic_block(feature),
                         "bloques_modelado": model_blocks(feature),
                         "primera_observacion": str(dates.min()) if len(dates) else None,
                         "ultima_observacion": str(dates.max()) if len(dates) else None,
                         "cobertura_total_pct": 100 * values.notna().mean(),
                         "cobertura_desarrollo_pct": 100 * values.loc[dev].notna().mean(),
                         "cobertura_train_inicial_pct": 100 * values.loc[train].notna().mean(),
                         "n_observados": int(values.notna().sum())})
        for year, indices in data.groupby(data["quarter"].dt.year).groups.items():
            by_year.append({"variable": feature, "bloque": economic_block(feature), "year": year,
                            "cobertura_pct": 100 * values.loc[indices].notna().mean()})
        for scope, mask in (("desarrollo", dev), ("total_descriptivo", pd.Series(True, index=data.index))):
            clean = values.loc[mask].dropna()
            q = clean.quantile([0, .01, .25, .5, .75, .99, 1])
            iqr = q.loc[.75] - q.loc[.25]
            extreme = ((clean < q.loc[.25] - 3 * iqr) | (clean > q.loc[.75] + 3 * iqr)).sum()
            quantiles.append({"variable": feature, "ambito": scope, "n": len(clean),
                              "min": q.loc[0], "p01": q.loc[.01], "p25": q.loc[.25],
                              "mediana": q.loc[.5], "p75": q.loc[.75], "p99": q.loc[.99],
                              "max": q.loc[1], "n_fuera_3_iqr": int(extreme)})
    for province, group in data.groupby("province_code", sort=True):
        available = group.loc[group[PRICE].notna(), "quarter"]
        all_quarters = pd.period_range(group["quarter"].min(), group["quarter"].max(), freq="Q")
        absent = all_quarters.difference(pd.PeriodIndex(group["quarter"]))
        internal = (pd.period_range(available.min(), available.max(), freq="Q")
                    .difference(pd.PeriodIndex(available)) if len(available) else pd.PeriodIndex([], freq="Q"))
        calendars.append({"province_code": province,
                          "province": group["province"].iloc[0] if "province" in group else province,
                          "n_filas": len(group), "n_precios_observados": len(available),
                          "primer_precio": str(available.min()) if len(available) else None,
                          "ultimo_precio": str(available.max()) if len(available) else None,
                          "n_filas_calendario_ausentes": len(absent),
                          "n_huecos_internos_precio": len(internal),
                          "trimestres_sin_precio_internos": ";".join(map(str, internal))})
    return {"cobertura_variables": pd.DataFrame(coverage),
            "cobertura_anual": pd.DataFrame(by_year),
            "distribuciones": pd.DataFrame(quantiles),
            "auditoria_calendario_provincial": pd.DataFrame(calendars)}


def annual_evolution(panel: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    data = normalize_panel(panel)
    lookup = data.set_index(["province_code", "quarter"])[PRICE]
    keys = pd.MultiIndex.from_arrays([data["province_code"], data["quarter"] - 4])
    previous = lookup.reindex(keys).to_numpy(dtype=float, na_value=np.nan)
    current = data[PRICE].to_numpy(dtype=float, na_value=np.nan)
    valid = (current > 0) & (previous > 0)
    growth = np.full(len(data), np.nan)
    growth[valid] = 100 * np.log(current[valid] / previous[valid])
    data["crecimiento_log_interanual"] = growth
    quarterly = data.groupby("quarter").agg(
        n_provincias=(PRICE, "count"), precio_mediana=(PRICE, "median"),
        precio_p10=(PRICE, lambda x: x.quantile(.1)), precio_p90=(PRICE, lambda x: x.quantile(.9)),
        crecimiento_mediana=("crecimiento_log_interanual", "median"),
        crecimiento_p10=("crecimiento_log_interanual", lambda x: x.quantile(.1)),
        crecimiento_p90=("crecimiento_log_interanual", lambda x: x.quantile(.9))).reset_index()
    annual = data.groupby(data["quarter"].dt.year.rename("year")).agg(
        n_precios=(PRICE, "count"), precio_mediana=(PRICE, "median"),
        crecimiento_log_mediana=("crecimiento_log_interanual", "median"),
        crecimiento_log_p10=("crecimiento_log_interanual", lambda x: x.quantile(.1)),
        crecimiento_log_p90=("crecimiento_log_interanual", lambda x: x.quantile(.9))).reset_index()
    return quarterly, annual


def save_figures(reports: dict[str, pd.DataFrame], output_dir: Path,
                 cutoff: str = DEVELOPMENT_END) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    coverage = reports["cobertura_anual"].pivot(index="variable", columns="year", values="cobertura_pct")
    fig, ax = plt.subplots(figsize=(13, max(7, .20 * len(coverage))))
    shown = ax.imshow(coverage.to_numpy(), aspect="auto", vmin=0, vmax=100, cmap="YlGnBu")
    ax.set_yticks(range(len(coverage)), coverage.index, fontsize=7)
    ax.set_xticks(range(0, len(coverage.columns), 2), coverage.columns[::2], rotation=45)
    ax.set_title("Cobertura anual observada; diagnóstico descriptivo de toda la muestra")
    fig.colorbar(shown, ax=ax, label="Observaciones disponibles (%)")
    fig.tight_layout()
    fig.savefig(output_dir / "cobertura_anual.png", dpi=150)
    plt.close(fig)

    quarterly = reports["evolucion_trimestral"]
    dates = pd.PeriodIndex(quarterly["quarter"], freq="Q").to_timestamp()
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    for ax, prefix, ylabel in zip(axes, ("precio", "crecimiento"), ("Valor tasado (€/m²)", "100 log(P[t]/P[t−4])")):
        ax.fill_between(dates, quarterly[f"{prefix}_p10"], quarterly[f"{prefix}_p90"], alpha=.2, label="P10–P90 entre provincias")
        ax.plot(dates, quarterly[f"{prefix}_mediana"], label="Mediana provincial", color="#165880")
        ax.axvline(pd.Period(cutoff).end_time, color="#9d4434", ls="--", label="Fin de desarrollo")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=.15)
    axes[0].legend(loc="upper left", fontsize=8)
    axes[1].axhline(0, color="gray", lw=.7)
    axes[0].set_title("Evolución descriptiva y dispersión territorial (sin ponderación; no es un índice nacional)")
    fig.tight_layout()
    fig.savefig(output_dir / "evolucion_precio_crecimiento.png", dpi=180)
    plt.close(fig)

    correlations = reports["correlaciones_crecimiento"]
    eligible = correlations.loc[(correlations["n_pares"] >= 200)
                                & (correlations["cobertura_etiquetados_pct"] >= 30)]
    if eligible.empty:
        return
    best = eligible.assign(abs_rho=eligible["spearman_sin_media_provincial"].abs()).groupby("variable")["abs_rho"].max().nlargest(25).index
    matrix = eligible.loc[eligible["variable"].isin(best)].pivot(
        index="variable", columns="horizonte", values="spearman_sin_media_provincial").reindex(best)
    fig, ax = plt.subplots(figsize=(8, 10))
    shown = ax.imshow(matrix.to_numpy(), aspect="auto", vmin=-1, vmax=1, cmap="RdBu_r")
    ax.set_yticks(range(len(matrix)), matrix.index, fontsize=8)
    ax.set_xticks(range(len(matrix.columns)), [f"h={h}" for h in matrix.columns])
    ax.set_title(f"Asociación con crecimiento futuro hasta {cutoff}\nSpearman tras restar medias provinciales; no implica capacidad predictiva")
    fig.colorbar(shown, ax=ax, label="Spearman")
    fig.tight_layout()
    fig.savefig(output_dir / "correlaciones_crecimiento.png", dpi=180)
    plt.close(fig)


def main(input_path: str | Path = DEFAULT_INPUT, output_dir: str | Path = DEFAULT_OUTPUT,
         cutoff: str = DEVELOPMENT_END) -> dict[str, pd.DataFrame]:
    input_path, output_dir = Path(input_path), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    panel = normalize_panel(pd.read_parquet(input_path))
    reports = {**coverage_reports(panel, cutoff), **selection_reports(panel, cutoff)}
    quarterly, annual = annual_evolution(panel)
    reports.update(evolucion_trimestral=quarterly, evolucion_anual=annual)
    for name, frame in reports.items():
        frame.to_csv(output_dir / f"{name}.csv", index=False, encoding="utf-8")
    save_figures(reports, output_dir / "figuras", cutoff)
    metadata = {
        "input": str(input_path.resolve()),
        "input_sha256": hashlib.sha256(input_path.read_bytes()).hexdigest(),
        "development_end_origin_and_target": cutoff,
        "initial_training_coverage_end": INITIAL_TRAIN_END,
        "horizons": list(HORIZONS), "random_state_mutual_information": 42,
        "model_feature_blocks": FEATURE_BLOCKS,
        "selection_scope": "Correlaciones, redundancia y MI: solo orígenes y etiquetas hasta el corte.",
        "descriptive_scope": "Cobertura, distribuciones total_descriptivo y evolución incluyen todos los datos; no seleccionan modelos.",
        "within_province_correlation": "Se resta la media provincial de X e Y calculada sobre cada muestra de pares y se aplica Spearman.",
        "regime_definition": "Ventanas descriptivas prefijadas; tanto origen como objetivo pertenecen a la ventana.",
        "external_feature_lags": "Retardos adicionales 0, 1 y 4 sobre las variables disponibles en t; se buscan fechas exactas.",
        "mutual_information": "Diagnóstico univariante in-sample; imputación mediana solo en desarrollo etiquetado; ausencias pueden inducir asociación.",
        "outliers": "Se identifican valores fuera de 3 IQR; no se eliminan ni winsorizan.",
        "limitations": [
            "Correlación no demuestra causalidad ni mejora predictiva fuera de muestra.",
            "Las medias provinciales no eliminan tendencias comunes ni dependencia temporal.",
            "Variables con cobertura distinta se calculan sobre muestras distintas; consultar n_pares y estabilidad.",
            "Ratios que incluyen precio y retardos mantienen una asociación mecánica con el nivel.",
            "Los periodos recientes ya se inspeccionaron en fases anteriores; no constituyen un test ciego nuevo.",
            "La disponibilidad depende de los retrasos conservadores del panel; no hay vintages históricos completos.",
            "La importancia final se determina mediante ablaciones y validación temporal, no por este ranking."],
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"EDA: {len(reports)} tablas y figuras en {output_dir}")
    return reports


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--development-end", default=DEVELOPMENT_END)
    args = parser.parse_args()
    main(args.input, args.output_dir, args.development_end)
