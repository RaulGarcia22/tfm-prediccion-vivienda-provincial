"""Backtesting temporal provincial para horizontes de 1 a 12 trimestres.

Cada evaluación entrega al modelo exclusivamente la serie observada hasta la
fecha de origen. El modelo genera la trayectoria completa sin recibir los
valores reales intermedios posteriores al origen.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Protocol, runtime_checkable

import numpy as np
import pandas as pd


@runtime_checkable
class ForecastModel(Protocol):
    """Contrato mínimo que deben implementar los modelos provinciales."""

    name: str

    def fit(self, history: pd.Series) -> "ForecastModel":
        """Ajusta el modelo usando una serie que termina en el origen."""

    def forecast(self, steps: int) -> np.ndarray:
        """Devuelve una trayectoria de longitud ``steps`` sin datos futuros."""


class LastValueModel:
    """Baseline de último valor ya usado en los experimentos del proyecto."""

    name = "Baseline último precio"

    def __init__(self) -> None:
        self._last_value: float | None = None

    def fit(self, history: pd.Series) -> "LastValueModel":
        clean = history.dropna().astype(float)
        if clean.empty:
            raise ValueError("El modelo necesita al menos una observación válida.")
        self._last_value = float(clean.iloc[-1])
        return self

    def forecast(self, steps: int) -> np.ndarray:
        if self._last_value is None:
            raise RuntimeError("Debe ejecutar fit antes de forecast.")
        if steps < 1:
            raise ValueError("steps debe ser positivo.")
        return np.repeat(self._last_value, steps).astype(float)


def _as_quarter(value: object) -> pd.Period:
    return pd.Period(value, freq="Q")


def _validated_series(series: pd.Series) -> pd.Series:
    result = series.copy()
    result.index = pd.PeriodIndex(result.index, freq="Q")
    result = result.sort_index().astype(float)
    if result.index.has_duplicates:
        raise ValueError("La serie provincial contiene trimestres duplicados.")
    if result.empty:
        raise ValueError("La serie provincial está vacía.")
    return result.reindex(pd.period_range(result.index.min(), result.index.max(), freq="Q"))


def backtest_provincial_series(
    series: pd.Series,
    model: ForecastModel,
    origin: object,
    horizon: int = 12,
) -> pd.DataFrame:
    """Evalúa una trayectoria provincial desde un origen histórico.

    Parameters
    ----------
    series:
        Serie completa, indexada por trimestre. Los valores posteriores al
        origen se reservan exclusivamente para evaluación.
    model:
        Instancia nueva del modelo. ``fit`` solo recibe ``series[:origin]``.
    origin:
        Último trimestre que se considera observado.
    horizon:
        Número máximo de pasos trimestrales que se desea evaluar.

    Returns
    -------
    DataFrame con predicción, valor real y errores para cada horizonte cuyo
    valor real ya está disponible.
    """
    if horizon < 1:
        raise ValueError("horizon debe ser positivo.")
    values = _validated_series(series)
    origin_period = _as_quarter(origin)
    history = values.loc[values.index <= origin_period]
    if history.empty or history.index.max() != origin_period or pd.isna(history.iloc[-1]):
        raise ValueError(f"No existe un valor observado en el origen {origin_period}.")

    model.fit(history.copy())
    predictions = np.asarray(model.forecast(horizon), dtype=float).reshape(-1)
    if len(predictions) != horizon:
        raise ValueError(
            f"El modelo devolvió {len(predictions)} valores; se esperaban {horizon}."
        )
    if not np.isfinite(predictions).all():
        raise ValueError("El modelo generó predicciones no finitas.")

    rows: list[dict[str, object]] = []
    for step, prediction in enumerate(predictions, start=1):
        forecast_quarter = origin_period + step
        if forecast_quarter not in values.index or pd.isna(values.loc[forecast_quarter]):
            continue
        actual = float(values.loc[forecast_quarter])
        error = float(prediction - actual)
        rows.append({
            "model": model.name,
            "origin": origin_period,
            "forecast_quarter": forecast_quarter,
            "horizon": step,
            "prediction": float(prediction),
            "actual": actual,
            "error": error,
            "absolute_error": abs(error),
            "squared_error": error**2,
            "ape_pct": np.nan if actual == 0 else 100 * abs(error) / abs(actual),
        })
    return pd.DataFrame(rows)


def run_multihorizon_backtest(
    panel: pd.DataFrame,
    model_factory: Callable[[], ForecastModel],
    origins: Iterable[object],
    horizon: int = 12,
    *,
    province_column: str = "province_code",
    quarter_column: str = "quarter",
    value_column: str = "valor_tasado_m2",
    province_name_column: str = "province",
) -> pd.DataFrame:
    """Ejecuta el mismo backtest para todas las provincias y orígenes."""
    required = {province_column, quarter_column, value_column}
    missing = required.difference(panel.columns)
    if missing:
        raise KeyError(f"Faltan columnas requeridas: {sorted(missing)}")

    data = panel.copy()
    data[quarter_column] = pd.PeriodIndex(data[quarter_column], freq="Q")
    data = data.sort_values([province_column, quarter_column])
    if data.duplicated([province_column, quarter_column]).any():
        raise ValueError("El panel contiene claves provincia-trimestre duplicadas.")

    origin_periods = [_as_quarter(origin) for origin in origins]
    results: list[pd.DataFrame] = []
    for province_code, group in data.groupby(province_column, sort=True):
        series = group.set_index(quarter_column)[value_column]
        province_name = (
            str(group[province_name_column].dropna().iloc[0])
            if province_name_column in group and group[province_name_column].notna().any()
            else str(province_code)
        )
        available = set(series.dropna().index)
        for origin in origin_periods:
            if origin not in available:
                continue
            result = backtest_provincial_series(
                series=series,
                model=model_factory(),
                origin=origin,
                horizon=horizon,
            )
            if result.empty:
                continue
            result.insert(1, "province_code", str(province_code).zfill(2))
            result.insert(2, "province", province_name)
            results.append(result)

    columns = [
        "model", "province_code", "province", "origin", "forecast_quarter",
        "horizon", "prediction", "actual", "error", "absolute_error",
        "squared_error", "ape_pct",
    ]
    if not results:
        return pd.DataFrame(columns=columns)
    return pd.concat(results, ignore_index=True)[columns]


def _metric_row(group: pd.DataFrame) -> pd.Series:
    return pd.Series({
        "observations": len(group),
        "MAE": group["absolute_error"].mean(),
        "RMSE": np.sqrt(group["squared_error"].mean()),
        "MAPE_pct": group["ape_pct"].mean(),
        "mean_error": group["error"].mean(),
    })


def metrics_by_horizon(results: pd.DataFrame) -> pd.DataFrame:
    """Calcula métricas homogéneas para cada modelo y horizonte."""
    if results.empty:
        return pd.DataFrame(
            columns=["model", "horizon", "observations", "MAE", "RMSE", "MAPE_pct", "mean_error"]
        )
    return (
        results.groupby(["model", "horizon"], as_index=False, sort=True)
        .apply(_metric_row, include_groups=False)
        .reset_index(drop=True)
    )


def horizon_band(horizon: pd.Series) -> pd.Categorical:
    """Clasifica horizontes en corto, medio y largo plazo."""
    return pd.cut(
        horizon,
        bins=[0, 4, 8, 12],
        labels=["corto_1_4", "medio_5_8", "largo_9_12"],
        include_lowest=True,
    )


def metrics_by_band(results: pd.DataFrame) -> pd.DataFrame:
    """Agrega métricas para los bloques 1-4, 5-8 y 9-12."""
    if results.empty:
        return pd.DataFrame(
            columns=["model", "horizon_band", "observations", "MAE", "RMSE", "MAPE_pct", "mean_error"]
        )
    data = results.copy()
    data["horizon_band"] = horizon_band(data["horizon"])
    return (
        data.groupby(["model", "horizon_band"], as_index=False, observed=True, sort=True)
        .apply(_metric_row, include_groups=False)
        .reset_index(drop=True)
    )


def origin_coverage(
    results: pd.DataFrame,
    requested_origins: Iterable[object],
    province_count: int,
    horizon: int,
) -> pd.DataFrame:
    """Documenta cuántos horizontes pueden evaluarse en cada origen."""
    origins = pd.PeriodIndex([_as_quarter(value) for value in requested_origins], freq="Q")
    rows = []
    for origin in origins:
        part = results.loc[results["origin"].eq(origin)] if not results.empty else results
        rows.append({
            "origin": origin,
            "requested_horizon": horizon,
            "max_evaluable_horizon": int(part["horizon"].max()) if not part.empty else 0,
            "provinces_evaluated": int(part["province_code"].nunique()) if not part.empty else 0,
            "expected_provinces": province_count,
            "observations": len(part),
        })
    return pd.DataFrame(rows)
