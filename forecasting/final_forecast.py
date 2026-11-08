"""Aplica la selección congelada a una única fecha de pronóstico.

Esta etapa no compara candidatos, recalibra bandas ni utiliza resultados de
evaluación para cambiar el modelo. Cada horizonte conserva su especificación
y vuelve a ajustarla con las etiquetas ya observadas.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from .calibration import apply_intervals
from .candidate_models import GlobalHuberGrowthModel
from .direct import ModelSpec
from .experiment import FORECAST_ORIGIN, FORECAST_END
from .unified import BASELINE_LABELS, predict_at_origin


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_final_forecast(frame: pd.DataFrame, provinces: list[str],
                            origin: object = FORECAST_ORIGIN, end: object = FORECAST_END) -> None:
    """Exige una fila por provincia y trimestre, precios y bandas válidos."""
    origin, end = pd.Period(origin, freq="Q"), pd.Period(end, freq="Q")
    steps = end.ordinal - origin.ordinal
    if steps < 1:
        raise ValueError("El fin de previsión debe ser posterior al origen.")
    expected = pd.MultiIndex.from_product(
        [sorted(provinces), range(1, steps + 1)], names=["province_code", "horizon"])
    actual = pd.MultiIndex.from_frame(frame[["province_code", "horizon"]])
    if actual.has_duplicates or set(actual) != set(expected):
        raise AssertionError("La previsión no contiene la cuadrícula provincial completa.")
    if not pd.PeriodIndex(frame.origin, freq="Q").equals(
        pd.PeriodIndex([origin] * len(frame), freq="Q")
    ):
        raise AssertionError("Origen distinto del origen de previsión fijado.")
    target = pd.PeriodIndex(frame.forecast_quarter, freq="Q")
    expected_target = pd.PeriodIndex([origin + int(h) for h in frame.horizon], freq="Q")
    if not target.equals(expected_target):
        raise AssertionError("El trimestre objetivo no coincide con origen + horizonte.")
    numeric = frame[["prediction", "lower_80", "upper_80", "lower_95", "upper_95"]]
    if not np.isfinite(numeric.to_numpy(dtype=float)).all() or not (numeric > 0).all().all():
        raise AssertionError("Los precios y límites deben ser positivos y finitos.")
    if not (frame.lower_95.le(frame.lower_80) & frame.lower_80.le(frame.prediction)
            & frame.prediction.le(frame.upper_80) & frame.upper_80.le(frame.upper_95)).all():
        raise AssertionError("Intervalos invertidos o niveles de confianza incoherentes.")
    if frame.selection_id.isna().any() or frame.selection_id.nunique() != 1:
        raise AssertionError("Todas las previsiones deben proceder de una única selección.")
    if frame.groupby("horizon").candidate_id.nunique().ne(1).any():
        raise AssertionError("Un horizonte mezcla especificaciones seleccionadas.")


def forecast_selected(panel: pd.DataFrame, selection: dict, calibration: pd.DataFrame,
                      origin: object = FORECAST_ORIGIN, end: object = FORECAST_END
                      ) -> tuple[pd.DataFrame, list[dict]]:
    """Ajusta los candidatos de fase 1 y aplica exclusivamente bandas congeladas.

    Los hashes y la procedencia de selección/calibración deben verificarse con
    stages.load_selection/load_calibration antes de invocar esta función.
    Se corta físicamente el panel antes de construir cualquier diseño.
    """
    origin, end = pd.Period(origin, freq="Q"), pd.Period(end, freq="Q")
    steps = end.ordinal - origin.ordinal
    if steps < 1:
        raise ValueError("El fin de previsión debe ser posterior al origen.")
    data = panel.copy()
    data["quarter"] = pd.PeriodIndex(data.quarter, freq="Q")
    data = data.loc[data.quarter.le(origin)].copy()
    if data.empty or data.quarter.max() != origin:
        raise ValueError(f"El panel no contiene el origen {origin}.")
    data["province_code"] = data.province_code.astype(str).str.zfill(2)
    if data.duplicated(["province_code", "quarter"]).any():
        raise ValueError("Claves provinciales duplicadas.")
    provinces = sorted(data.province_code.unique())
    current = data.loc[data.quarter.eq(origin)]
    if set(current.province_code) != set(provinces) or not (
        np.isfinite(current.valor_tasado_m2) & current.valor_tasado_m2.gt(0)
    ).all():
        raise ValueError("Todas las provincias necesitan un precio positivo en el origen.")
    if not selection.get("selection_id"):
        raise ValueError("Falta el identificador de la selección congelada.")
    if not {str(h) for h in range(1, steps + 1)}.issubset(selection["selected_by_horizon"]):
        raise ValueError("Falta la selección de algún horizonte de previsión.")
    if set(calibration.confidence.unique()) != {.8, .95}:
        raise ValueError("Se requieren bandas calibradas al 80 y 95 por ciento.")
    if not np.isfinite(calibration.log_radius).all() or calibration.log_radius.lt(0).any():
        raise ValueError("Radios de calibración no válidos.")

    frames, audits = [], []
    for horizon in range(1, steps + 1):
        spec = ModelSpec.from_dict(selection["selected_by_horizon"][str(horizon)])
        detail, record = predict_at_origin(data, spec, horizon, origin, require_actual=False)
        if not detail.candidate_id.eq(spec.key).all():
            raise AssertionError("El modelo ajustado difiere de la selección congelada.")
        record.update(selection_id=selection["selection_id"], specification=spec.to_dict(),
                      forecast_target=str(origin + horizon), observed_data_end=str(origin),
                      province_count=len(provinces), categorical_features=["province_code"])
        if spec.family == "recursive_huber":
            training = GlobalHuberGrowthModel.training_frame(data, origin)
            record.update(n_train=len(training), training_target_max=str((training.quarter + 1).max()),
                          numeric_features=list(GlobalHuberGrowthModel.numeric_features),
                          excluded_features={}, training_horizon=1, prediction_strategy="recursive")
        elif spec.family in BASELINE_LABELS:
            record.update(n_train=None, training_target_max=None, numeric_features=["valor_tasado_m2"],
                          categorical_features=[], excluded_features={}, prediction_strategy="local_reference",
                          history_observations=int(data.valor_tasado_m2.notna().sum()))
        else:
            record.update(training_horizon=horizon, prediction_strategy="direct")
        if record.get("training_target_max") is not None and pd.Period(record["training_target_max"]) > origin:
            raise AssertionError("El ajuste final ha utilizado etiquetas posteriores al origen.")
        detail["selection_id"] = selection["selection_id"]
        frames.append(detail)
        audits.append(record)
    forecast = apply_intervals(pd.concat(frames, ignore_index=True), calibration)
    forecast = forecast.drop(columns=["actual", "error", "absolute_error", "squared_error", "ape_pct"])
    forecast = forecast.sort_values(["province_code", "horizon"]).reset_index(drop=True)
    validate_final_forecast(forecast, provinces, origin, end)
    return forecast, audits
