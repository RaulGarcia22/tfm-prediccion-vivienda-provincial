"""Fase 2: aplica la selección de fase 1, con origen fijo 2026T1, hasta 2028T4."""
from __future__ import annotations

import re
import unicodedata

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from forecasting.experiment import (
    ROOT, PANEL_PATH, REPORTS, SELECTION_PATH, FORECAST_ORIGIN as ORIGIN,
    FORECAST_END as END, write_json,
)
from forecasting.final_forecast import file_sha256, forecast_selected
from forecasting.stages import load_selection, load_calibration, validate_evaluation_manifest
from forecasting.unified import frame_digest, read_panel

INPUT = PANEL_PATH
OUTPUT = ROOT / "data" / "forecasts"
FIGURES = OUTPUT / "figures"
FORECAST_ORIGIN = pd.Period(ORIGIN, freq="Q")
FORECAST_END = pd.Period(END, freq="Q")
STEPS = FORECAST_END.ordinal - FORECAST_ORIGIN.ordinal
BASENAME = "prevision_provincial_2026T2_2028T4"


def slug(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", "_", normalized.lower()).strip("_")


def plot_province(history: pd.DataFrame, forecast: pd.DataFrame, code: str, province: str) -> None:
    observed = history.loc[
        history.province_code.eq(code)
        & history.quarter.between(pd.Period("2015Q1"), FORECAST_ORIGIN)
    ].sort_values("quarter")
    future = forecast.loc[forecast.province_code.eq(code)].sort_values("horizon")
    observed_dates = observed.quarter.dt.to_timestamp(how="end")
    future_dates = pd.PeriodIndex(future.forecast_quarter, freq="Q").to_timestamp(how="end")
    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.plot(observed_dates, observed.valor_tasado_m2, label="Histórico", color="#2878B5", lw=2)
    bridge_dates = pd.DatetimeIndex([FORECAST_ORIGIN.to_timestamp(how="end"), *future_dates.tolist()])
    bridge_values = np.r_[observed.iloc[-1].valor_tasado_m2, future.prediction.to_numpy()]
    ax.plot(bridge_dates, bridge_values, label="Modelos seleccionados por horizonte", color="#E07A3F", lw=2)
    for suffix, alpha in [(95, .15), (80, .28)]:
        ax.fill_between(future_dates, future[f"lower_{suffix}"].to_numpy(),
                        future[f"upper_{suffix}"].to_numpy(), color="#E07A3F", alpha=alpha,
                        label=f"Banda marginal nominal {suffix} %")
    ax.axvline(FORECAST_ORIGIN.to_timestamp(how="end"), color="grey", ls="--", lw=1)
    ax.set(title=f"Valor tasado — {province}\nPrevisión experimental con origen en 2026T1", xlabel="", ylabel="€/m²")
    ax.legend(fontsize=9)
    ax.grid(alpha=.25)
    fig.tight_layout()
    fig.savefig(FIGURES / f"{code}_{slug(province)}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_overview(history: pd.DataFrame, forecast: pd.DataFrame) -> None:
    # Una media de límites provinciales no es un intervalo de la media territorial.
    observed = history.loc[history.quarter.between(pd.Period("2015Q1"), FORECAST_ORIGIN)]
    aggregate = observed.groupby("quarter").valor_tasado_m2.agg(["mean", "count"])
    province_count = forecast.province_code.nunique()
    aggregate.loc[aggregate["count"].ne(province_count), "mean"] = np.nan
    future = forecast.groupby("forecast_quarter", sort=True).prediction.mean()
    observed_dates = aggregate.index.to_timestamp(how="end")
    future_dates = pd.PeriodIndex(future.index, freq="Q").to_timestamp(how="end")
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(observed_dates, aggregate["mean"], color="#2878B5", lw=2, label="Media provincial histórica")
    bridge_dates = pd.DatetimeIndex([FORECAST_ORIGIN.to_timestamp(how="end"), *future_dates.tolist()])
    bridge_values = np.r_[aggregate.loc[FORECAST_ORIGIN, "mean"], future.to_numpy()]
    ax.plot(bridge_dates, bridge_values, color="#E07A3F", lw=2, label="Media de previsiones provinciales")
    ax.axvline(FORECAST_ORIGIN.to_timestamp(how="end"), color="grey", ls="--", lw=1)
    ax.set(title="Previsión experimental: media equiponderada de 52 territorios; origen 2026T1\nNo representa un precio nacional ponderado",
           xlabel="", ylabel="€/m²")
    ax.legend()
    ax.grid(alpha=.25)
    fig.tight_layout()
    fig.savefig(FIGURES / "00_resumen_media_provincial.png", dpi=170, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    panel = read_panel(INPUT)
    selection = load_selection(panel)
    calibration = load_calibration(panel, selection)
    evaluation = validate_evaluation_manifest(panel, selection)
    interval_coverage = pd.read_csv(REPORTS / "interval_coverage.csv")
    metrics = pd.read_csv(REPORTS / "evaluation_metrics.csv")
    selected_mae = metrics.loc[metrics.role.eq("selected")].set_index("horizon").MAE
    reference_mae = metrics.loc[metrics.role.eq("annual_trend")].set_index("horizon").MAE
    point_comparison = pd.DataFrame({"selected_MAE": selected_mae, "annual_trend_MAE": reference_mae})
    point_comparison["MAE_improvement_vs_annual_trend"] = (
        point_comparison.annual_trend_MAE - point_comparison.selected_MAE)
    forecast_point_comparison = point_comparison.reindex(range(1, STEPS + 1))
    forecast_coverage = interval_coverage.loc[interval_coverage.horizon.between(1, STEPS)]
    point_passed = bool(forecast_point_comparison.notna().all().all()
                        and forecast_point_comparison.MAE_improvement_vs_annual_trend.gt(0).all())
    coverage_passed = bool(len(forecast_coverage) == STEPS * 2
                           and forecast_coverage.empirical_coverage.ge(forecast_coverage.nominal_coverage).all())
    print(f"[FIT] Selección {selection['selection_id'][:12]}, origen fijo {FORECAST_ORIGIN}", flush=True)
    forecast, audit = forecast_selected(panel, selection, calibration)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    final = forecast.rename(columns={
        "province": "provincia", "forecast_quarter": "trimestre", "prediction": "prediccion",
        "lower_80": "limite_inferior_80", "upper_80": "limite_superior_80",
        "lower_95": "limite_inferior_95", "upper_95": "limite_superior_95",
    })[["province_code", "provincia", "origin", "trimestre", "horizon", "prediccion",
        "limite_inferior_80", "limite_superior_80", "limite_inferior_95", "limite_superior_95",
        "candidate_id", "model", "family", "feature_set", "selection_id"]]
    csv_path, parquet_path = OUTPUT / f"{BASENAME}.csv", OUTPUT / f"{BASENAME}.parquet"
    final.to_csv(csv_path, index=False)
    final.to_parquet(parquet_path, index=False)
    audit_path = OUTPUT / f"{BASENAME}_fit_audit.json"
    write_json(audit_path, audit)
    sources = [
        INPUT, SELECTION_PATH, REPORTS / "interval_calibration.csv",
        REPORTS / "calibration_manifest.json", REPORTS / "evaluation_manifest.json",
        REPORTS / "interval_coverage.csv", REPORTS / "evaluation_metrics.csv",
        ROOT / "data/reports/data_build_manifest.json", ROOT / "data/reports/source_availability_audit.csv",
        ROOT / "sources.json", ROOT / "09_generar_prevision_final.py",
        ROOT / "forecasting/final_forecast.py", ROOT / "forecasting/stages.py",
        ROOT / "forecasting/calibration.py",
    ]
    configuration = {
        "selection_id": selection["selection_id"],
        "model": "especificación congelada por horizonte de fase 1",
        "selected_by_horizon": {str(h): selection["selected_by_horizon"][str(h)] for h in range(1, STEPS + 1)},
        "selection_implementation_hash": selection["implementation_hash"],
        "training_data_end": str(FORECAST_ORIGIN), "forecast_origin": str(FORECAST_ORIGIN),
        "forecast_start": str(FORECAST_ORIGIN + 1), "forecast_end": str(FORECAST_END),
        "steps": STEPS, "province_count": final.province_code.nunique(), "forecast_rows": len(final),
        "expected_rows": final.province_code.nunique() * STEPS,
        "deployment_status": "frozen_research_forecast",
        "operational_validation_passed": point_passed and coverage_passed,
        "operational_validation_diagnostics": {
            "interpretation": "Comprobación descriptiva posterior a la selección; no modifica candidatos ni demuestra validación prospectiva.",
            "point_rule": "MAE del seleccionado menor que tendencia interanual en todos los horizontes de producción",
            "interval_rule": "cobertura empírica al menos nominal en ambos niveles y todos los horizontes de producción",
            "point_rule_passed": point_passed, "interval_rule_passed": coverage_passed,
            "MAE_by_horizon": point_comparison.reset_index().to_dict(orient="records"),
        },
        "available_data_hash": frame_digest(panel.loc[panel.quarter.le(FORECAST_ORIGIN)]),
        "as_of_statement": "Previsión condicionada al panel hasta 2026T1; no se afirma usar los datos más recientes a la fecha de ejecución.",
        "refit_rule": "misma especificación y transformaciones que fase 1; etiquetas de entrenamiento <= 2026T1",
        "selection_rule": "selección congelada en desarrollo; evaluación posterior no cambia los ganadores",
        "interval_method": "pooled_absolute_log_error_finite_sample_rank",
        "nominal_confidences": [.8, .95], "calibration_target_end": str(calibration.target_end.max()),
        "interval_limitations": [
            "Bandas marginales por horizonte: no garantizan cobertura simultánea de la trayectoria.",
            "La dependencia temporal y espacial impide garantizar cobertura nominal exacta.",
            "La calibración histórica puede perder cobertura cuando cambia la dinámica del mercado.",
        ],
        "evaluation_target_start": evaluation["target_min"], "evaluation_target_end": evaluation["target_max"],
        "evaluation_status": "retrospectiva, con selección y calibración congeladas; historia ya examinada anteriormente",
        "evaluation_interval_coverage_by_horizon": interval_coverage.to_dict(orient="records"),
        "overview_definition": "media equiponderada de 52 provincias/ciudades autónomas; no precio nacional ponderado ni intervalo nacional",
        "trajectory_limit": "Los modelos directos se seleccionan por horizonte; la trayectoria no se suaviza después de la evaluación.",
        "source_files": {str(path.relative_to(ROOT)).replace("\\", "/"): file_sha256(path) for path in sources},
        "output_files": {str(path.relative_to(ROOT)).replace("\\", "/"): file_sha256(path)
                         for path in [csv_path, parquet_path, audit_path]},
    }
    write_json(OUTPUT / f"{BASENAME}_config.json", configuration)
    print("[PLOT] Figuras provinciales y media equiponderada", flush=True)
    for code, province in forecast[["province_code", "province"]].drop_duplicates().itertuples(index=False):
        plot_province(panel, forecast, code, province)
    plot_overview(panel, forecast)
    print(f"[OK] {csv_path.relative_to(ROOT)}: {len(final)} filas; {len(audit)} ajustes")
    print(f"[OK] {final.province_code.nunique() + 1} figuras generadas en {FIGURES.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
