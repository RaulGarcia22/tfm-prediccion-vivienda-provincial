"""Cuatro candidatos finales para el forecasting provincial del TFM."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import HuberRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from statsmodels.tsa.holtwinters import Holt

from .multihorizon import LastValueModel
from .calendar import complete_quarterly_grid


class PersistenceModel(LastValueModel):
    """Alias explícito del baseline de último valor."""

    name = "Persistencia"


class DampedRecentTrendModel:
    """Extrapola el crecimiento logarítmico reciente con pendiente amortiguada.

    La pendiente es la mediana del crecimiento trimestral logarítmico de los últimos
    ``lookback`` cambios. En el paso h se multiplica por ``damping ** (h-1)``.
    Los parámetros se fijan antes del backtesting y no se optimizan por origen.
    """

    name = "Tendencia amortiguada"

    def __init__(self, lookback: int = 8, damping: float = 0.85) -> None:
        if lookback < 1 or not 0 < damping < 1:
            raise ValueError("lookback debe ser positivo y damping debe estar entre 0 y 1.")
        self.lookback = lookback
        self.damping = damping
        self._last_log: float | None = None
        self._slope: float | None = None

    def fit(self, history: pd.Series) -> "DampedRecentTrendModel":
        clean = history.astype(float).sort_index()
        clean = clean.reindex(pd.period_range(clean.index.min(), clean.index.max(), freq="Q"))
        if len(clean) < 2 or (clean <= 0).any():
            raise ValueError("La tendencia amortiguada necesita dos precios positivos.")
        changes = np.log(clean).diff().iloc[-self.lookback:].dropna().to_numpy()
        if len(changes) == 0 or pd.isna(clean.iloc[-1]):
            raise ValueError("No hay cambios trimestrales recientes observados.")
        self._last_log = float(np.log(clean.iloc[-1]))
        self._slope = float(np.median(changes[-self.lookback :]))
        return self

    def forecast(self, steps: int) -> np.ndarray:
        if self._last_log is None or self._slope is None:
            raise RuntimeError("Debe ejecutar fit antes de forecast.")
        horizons = np.arange(1, steps + 1, dtype=float)
        accumulated = (1 - self.damping**horizons) / (1 - self.damping)
        return np.exp(self._last_log + self._slope * accumulated)


class HoltDampedModel:
    """Holt exponencial con tendencia aditiva amortiguada."""

    name = "Holt amortiguado"

    def __init__(self, damping_trend: float | None = None) -> None:
        self.damping_trend = damping_trend
        self._fit = None

    def fit(self, history: pd.Series) -> "HoltDampedModel":
        clean = history.astype(float).sort_index()
        clean = clean.reindex(pd.period_range(clean.index.min(), clean.index.max(), freq="Q"))
        missing = np.flatnonzero(clean.isna().to_numpy())
        if len(missing):
            clean = clean.iloc[missing[-1] + 1:]
        if len(clean) < 8:
            raise ValueError("Holt amortiguado necesita al menos ocho observaciones.")
        model = Holt(
            clean.to_numpy(),
            damped_trend=True,
            initialization_method="estimated",
        )
        kwargs = {"optimized": True, "remove_bias": False}
        if self.damping_trend is not None:
            kwargs["damping_trend"] = self.damping_trend
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            self._fit = model.fit(**kwargs)
        return self

    def forecast(self, steps: int) -> np.ndarray:
        if self._fit is None:
            raise RuntimeError("Debe ejecutar fit antes de forecast.")
        return np.asarray(self._fit.forecast(steps), dtype=float)


class GlobalHuberGrowthModel:
    """Huber global que predice crecimiento intertrimestral recursivamente."""

    name = "Huber global crecimiento"
    numeric_features = ["growth_t", "growth_lag1", "growth_lag3", "yoy_t", "next_quarter"]

    def __init__(self, epsilon: float = 1.35, alpha: float = 0.0001) -> None:
        self.epsilon = epsilon
        self.alpha = alpha
        self._model: Pipeline | None = None
        self._histories: dict[str, list[float]] = {}
        self._province_names: dict[str, str] = {}
        self._origin: pd.Period | None = None

    @staticmethod
    def _prepare_panel(panel: pd.DataFrame) -> pd.DataFrame:
        required = {"province_code", "quarter", "valor_tasado_m2"}
        missing = required.difference(panel.columns)
        if missing:
            raise KeyError(f"Faltan columnas para Huber global: {sorted(missing)}")
        data = panel.copy()
        data["quarter"] = pd.PeriodIndex(data["quarter"], freq="Q")
        data["province_code"] = data["province_code"].astype(str).str.zfill(2)
        return complete_quarterly_grid(data).sort_values(["province_code", "quarter"])

    @classmethod
    def training_frame(cls, panel: pd.DataFrame, origin: object) -> pd.DataFrame:
        """Construye ejemplos cuyo crecimiento objetivo ya se conoce al origen."""
        cutoff = pd.Period(origin, freq="Q")
        data = cls._prepare_panel(panel)
        data = data.loc[data["quarter"] <= cutoff].copy()
        if (data["valor_tasado_m2"] <= 0).any():
            raise ValueError("Huber de crecimiento requiere precios positivos.")
        grouped = data.groupby("province_code", sort=False)["valor_tasado_m2"]
        log_price = np.log(data["valor_tasado_m2"].astype(float))
        data["growth_t"] = log_price.groupby(data["province_code"]).diff() * 100
        data["growth_lag1"] = data.groupby("province_code")["growth_t"].shift(1)
        data["growth_lag3"] = data.groupby("province_code")["growth_t"].shift(3)
        data["yoy_t"] = grouped.transform(lambda values: np.log(values).diff(4) * 100)
        data["target_growth"] = data.groupby("province_code")["growth_t"].shift(-1)
        data["next_quarter"] = (data["quarter"] + 1).dt.quarter
        return data.dropna(subset=["target_growth", "valor_tasado_m2"])

    def fit(self, panel: pd.DataFrame, origin: object) -> "GlobalHuberGrowthModel":
        origin_period = pd.Period(origin, freq="Q")
        data = self._prepare_panel(panel)
        observed = data.loc[data["quarter"] <= origin_period].copy()
        last_by_province = observed.groupby("province_code")["quarter"].max()
        if not last_by_province.eq(origin_period).all() or observed.loc[
            observed.quarter.eq(origin_period), "valor_tasado_m2"
        ].isna().any():
            raise ValueError("Todas las provincias deben tener precio en el origen.")
        train = self.training_frame(observed, origin_period)
        numeric = Pipeline([
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            ("scaler", StandardScaler()),
        ])
        preprocess = ColumnTransformer([
            ("numeric", numeric, self.numeric_features),
            ("province", OneHotEncoder(handle_unknown="ignore", sparse_output=False), ["province_code"]),
        ], sparse_threshold=0)
        self._model = Pipeline([
            ("preprocess", preprocess),
            ("model", HuberRegressor(
                epsilon=self.epsilon, alpha=self.alpha, max_iter=2_000
            )),
        ])
        columns = self.numeric_features + ["province_code"]
        self._model.fit(train[columns], train["target_growth"])
        self._histories = {
            code: group["valor_tasado_m2"].astype(float).tolist()
            for code, group in observed.groupby("province_code", sort=True)
        }
        if "province" in observed:
            self._province_names = observed.groupby("province_code")["province"].last().astype(str).to_dict()
        else:
            self._province_names = {code: code for code in self._histories}
        self._origin = origin_period
        return self

    @staticmethod
    def _features(history: list[float], next_quarter: int) -> dict[str, float]:
        logs = np.log(np.asarray(history, dtype=float))
        changes = np.diff(logs) * 100
        return {
            "growth_t": float(changes[-1]) if len(changes) >= 1 else np.nan,
            "growth_lag1": float(changes[-2]) if len(changes) >= 2 else np.nan,
            "growth_lag3": float(changes[-4]) if len(changes) >= 4 else np.nan,
            "yoy_t": float((logs[-1] - logs[-5]) * 100) if len(logs) >= 5 else np.nan,
            "next_quarter": next_quarter,
        }

    def forecast(self, steps: int) -> pd.DataFrame:
        if self._model is None or self._origin is None:
            raise RuntimeError("Debe ejecutar fit antes de forecast.")
        histories = {code: values.copy() for code, values in self._histories.items()}
        rows: list[dict[str, object]] = []
        columns = self.numeric_features + ["province_code"]
        for horizon in range(1, steps + 1):
            target_quarter = self._origin + horizon
            features = []
            codes = sorted(histories)
            for code in codes:
                row = self._features(histories[code], target_quarter.quarter)
                row["province_code"] = code
                features.append(row)
            frame = pd.DataFrame(features, columns=columns)
            growth = self._model.predict(frame)
            for code, predicted_growth in zip(codes, growth):
                prediction = histories[code][-1] * np.exp(float(predicted_growth) / 100)
                histories[code].append(float(prediction))
                rows.append({
                    "model": self.name,
                    "province_code": code,
                    "province": self._province_names[code],
                    "origin": self._origin,
                    "forecast_quarter": target_quarter,
                    "horizon": horizon,
                    "prediction": float(prediction),
                })
        return pd.DataFrame(rows)


def backtest_global_huber(
    panel: pd.DataFrame,
    origins: list[object],
    horizon: int = 12,
) -> pd.DataFrame:
    """Evalúa Huber global conservando el mismo esquema que los modelos locales."""
    data = GlobalHuberGrowthModel._prepare_panel(panel)
    actual = data[["province_code", "quarter", "valor_tasado_m2"]].rename(
        columns={"quarter": "forecast_quarter", "valor_tasado_m2": "actual"}
    )
    results = []
    for origin in origins:
        model = GlobalHuberGrowthModel().fit(data, origin)
        prediction = model.forecast(horizon).merge(
            actual, on=["province_code", "forecast_quarter"], how="left", validate="many_to_one"
        ).dropna(subset=["actual"])
        prediction["error"] = prediction["prediction"] - prediction["actual"]
        prediction["absolute_error"] = prediction["error"].abs()
        prediction["squared_error"] = prediction["error"] ** 2
        prediction["ape_pct"] = 100 * prediction["absolute_error"] / prediction["actual"].abs()
        results.append(prediction)
    return pd.concat(results, ignore_index=True)
