"""Modelos directos por horizonte con información disponible en el origen.

Cada modelo estima 100 * log(P[t+h] / P[t]). Los predictores externos se
toman exclusivamente de la fila t: no necesitan conocer ni proyectar su
trayectoria futura. La selección temporal corresponde al evaluador, mientras
que este módulo aplica el embargo de las etiquetas también al ajuste final.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any
import warnings

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, HuberRegressor, LinearRegression, Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor
from threadpoolctl import threadpool_limits


AR_FEATURES = [
    "price_log", "growth_1q", "growth_4q", "growth_lag1", "growth_lag4",
    "volatility_4q", "quarter_sin", "quarter_cos",
]
EXTERNAL_BLOCKS = {
    "credit": [
        "hipotecas_numero_por_1000_hab", "importe_medio_hipoteca_eur",
        "euribor_12m", "compraventas_por_1000_hab",
    ],
    "economy": [
        "tasa_paro", "ipc_yoy", "renta_neta_media_persona",
        "renta_neta_media_hogar", "pib_per_capita_eur",
        "empleo_sobre_poblacion_pct",
    ],
    "demography": [
        "densidad_poblacion", "pct_poblacion_18_34", "pct_poblacion_25_39",
        "crecimiento_poblacion_18_34_yoy", "crecimiento_poblacion_25_39_yoy",
    ],
    "tourism": [
        "viajeros_hoteles_por_1000_hab", "pernoctaciones_hoteles_por_1000_hab",
    ],
    "supply": [
        "iniciadas_libre_por_1000_hab", "iniciadas_protegida_por_1000_hab",
        "terminadas_libre_por_1000_hab", "terminadas_protegida_por_1000_hab",
        "precio_suelo_m2", "peso_construccion_pib_pct",
    ],
}
# Todos los conjuntos incluyen la misma referencia autorregresiva.
FEATURE_BLOCKS = {"ar": list(AR_FEATURES)}
FEATURE_BLOCKS.update({key: AR_FEATURES + values for key, values in EXTERNAL_BLOCKS.items()})
FEATURE_BLOCKS["all"] = list(dict.fromkeys(
    AR_FEATURES + [column for block in EXTERNAL_BLOCKS.values() for column in block]
))

FAMILY_LABELS = {
    "ridge": "Ridge", "elasticnet": "Elastic Net", "huber": "Huber",
    "knn": "K vecinos", "random_forest": "Random Forest",
    "extra_trees": "Extra Trees", "hist_gradient_boosting": "HistGradientBoosting",
    "gradient_boosting": "Gradient Boosting", "adaboost": "AdaBoost",
    "mlp": "MLP", "linear": "Regresión lineal",
}


@dataclass(frozen=True)
class ModelSpec:
    """Especificación serializable; no contiene un estimador ya ajustado."""

    family: str
    feature_set: str
    params: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {"family": self.family, "feature_set": self.feature_set, "params": dict(self.params)}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ModelSpec":
        return cls(str(data["family"]), str(data["feature_set"]), dict(data.get("params", {})))

    @property
    def name(self) -> str:
        parameters = json.dumps(self.params, sort_keys=True, ensure_ascii=False)
        return f"{FAMILY_LABELS.get(self.family, self.family)} | {self.feature_set} | {parameters}"

    @property
    def key(self) -> str:
        payload = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]
        return f"{self.family}__{self.feature_set}__{digest}"


def candidate_specs() -> list[ModelSpec]:
    """Parrilla acotada de 93 configuraciones, fijada antes de evaluarlas.

    Las tres familias regularizadas prueban todas las ablaciones. El resto
    compara AR y el conjunto completo con dos configuraciones moderadas
    (OLS no tiene hiperparámetros de regularización).
    """
    specifications: list[ModelSpec] = []
    core = {
        "ridge": [{"alpha": value} for value in (0.1, 10.0, 100.0)],
        "elasticnet": [
            {"alpha": 0.01, "l1_ratio": 0.1},
            {"alpha": 0.01, "l1_ratio": 0.9},
            {"alpha": 0.1, "l1_ratio": 0.5},
        ],
        "huber": [
            {"epsilon": 1.35, "alpha": 0.0001},
            {"epsilon": 1.35, "alpha": 0.01},
            {"epsilon": 1.75, "alpha": 0.01},
        ],
    }
    for family, settings in core.items():
        for feature_set in FEATURE_BLOCKS:
            for params in settings:
                specifications.append(ModelSpec(family, feature_set, dict(params)))
    other = {
        "knn": [{"n_neighbors": 15}, {"n_neighbors": 40}],
        "random_forest": [
            {"min_samples_leaf": 5, "max_features": 0.7},
            {"min_samples_leaf": 15, "max_features": 1.0},
        ],
        "extra_trees": [
            {"min_samples_leaf": 5, "max_features": 0.7},
            {"min_samples_leaf": 15, "max_features": 1.0},
        ],
        "hist_gradient_boosting": [
            {"max_leaf_nodes": 7, "l2_regularization": 5.0},
            {"max_leaf_nodes": 15, "l2_regularization": 10.0},
        ],
        "gradient_boosting": [{"max_depth": 2}, {"max_depth": 3}],
        "adaboost": [{"learning_rate": 0.03}, {"learning_rate": 0.1}],
        "mlp": [
            {"hidden_layer_sizes": [24], "alpha": 0.1},
            {"hidden_layer_sizes": [32, 16], "alpha": 1.0},
        ],
        "linear": [{}],
    }
    for family, settings in other.items():
        for feature_set in ("ar", "all"):
            for params in settings:
                specifications.append(ModelSpec(family, feature_set, dict(params)))
    return specifications


def build_design(panel: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Construye variables y etiquetas mediante fechas exactas de calendario.

    Conserva las filas sin etiqueta para la predicción final. Si falta un
    trimestre intermedio, el retardo correspondiente queda ausente; jamás se
    sustituye por la observación anterior disponible.
    """
    if isinstance(horizon, bool) or not isinstance(horizon, (int, np.integer)) or horizon < 1:
        raise ValueError("horizon debe ser un número entero positivo de trimestres.")
    required = {"province_code", "quarter", "valor_tasado_m2"}
    missing = required.difference(panel.columns)
    if missing:
        raise KeyError(f"Faltan columnas para el diseño directo: {sorted(missing)}")
    frame = panel.copy()
    if frame["province_code"].isna().any() or frame["quarter"].isna().any():
        raise ValueError("Provincia y trimestre no pueden estar ausentes.")
    frame["province_code"] = frame["province_code"].astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(2)
    frame["quarter"] = pd.PeriodIndex(frame["quarter"], freq="Q")
    if frame.duplicated(["province_code", "quarter"]).any():
        raise ValueError("El panel contiene duplicados de provincia y trimestre.")
    frame = frame.sort_values(["province_code", "quarter"]).reset_index(drop=True)
    frame["valor_tasado_m2"] = pd.to_numeric(frame["valor_tasado_m2"], errors="coerce")
    values = frame["valor_tasado_m2"].where(
        np.isfinite(frame["valor_tasado_m2"]) & frame["valor_tasado_m2"].gt(0)
    )
    index = pd.MultiIndex.from_arrays([frame["province_code"], frame["quarter"]])
    price_lookup = pd.Series(values.to_numpy(), index=index)

    def calendar_price(offset: int) -> np.ndarray:
        lookup_index = pd.MultiIndex.from_arrays([
            frame["province_code"], frame["quarter"] + offset,
        ])
        return price_lookup.reindex(lookup_index).to_numpy(dtype=float)

    logs = {lag: np.log(calendar_price(-lag)) for lag in range(6)}
    frame["current_price"] = values
    frame["price_log"] = logs[0]
    frame["growth_1q"] = (logs[0] - logs[1]) * 100
    frame["growth_4q"] = (logs[0] - logs[4]) * 100
    frame["growth_lag1"] = (logs[1] - logs[2]) * 100
    frame["growth_lag4"] = (logs[4] - logs[5]) * 100
    recent_growth = np.column_stack([(logs[lag] - logs[lag + 1]) * 100 for lag in range(4)])
    frame["volatility_4q"] = np.std(recent_growth, axis=1, ddof=0)
    quarter_number = frame["quarter"].dt.quarter.to_numpy()
    frame["quarter_sin"] = np.sin(2 * np.pi * quarter_number / 4)
    frame["quarter_cos"] = np.cos(2 * np.pi * quarter_number / 4)
    frame["target_quarter"] = frame["quarter"] + horizon
    frame["target_price"] = calendar_price(horizon)
    frame["target_log_growth"] = (np.log(frame["target_price"]) - frame["price_log"]) * 100
    frame["horizon"] = int(horizon)
    for name in FEATURE_BLOCKS["all"]:
        if name in frame:
            frame[name] = pd.to_numeric(frame[name], errors="coerce").replace([np.inf, -np.inf], np.nan)
    return frame


def _make_estimator(spec: ModelSpec) -> Any:
    params = dict(spec.params)
    defaults: dict[str, Any]
    constructors = {
        "ridge": (Ridge, {}),
        "elasticnet": (ElasticNet, {"max_iter": 5_000, "tol": 1e-4, "random_state": 42}),
        "huber": (HuberRegressor, {"max_iter": 1_000, "tol": 1e-4}),
        "knn": (KNeighborsRegressor, {"weights": "distance", "n_jobs": 1}),
        "random_forest": (RandomForestRegressor, {"n_estimators": 80, "max_depth": 12, "n_jobs": 1, "random_state": 42}),
        "extra_trees": (ExtraTreesRegressor, {"n_estimators": 80, "max_depth": 12, "n_jobs": 1, "random_state": 42}),
        "hist_gradient_boosting": (HistGradientBoostingRegressor, {
            "max_iter": 100, "learning_rate": 0.06, "early_stopping": False, "random_state": 42,
        }),
        "gradient_boosting": (GradientBoostingRegressor, {
            "n_estimators": 80, "learning_rate": 0.05, "min_samples_leaf": 10,
            "loss": "huber", "random_state": 42,
        }),
        "adaboost": (AdaBoostRegressor, {
            "estimator": DecisionTreeRegressor(max_depth=3, min_samples_leaf=10, random_state=42),
            "n_estimators": 60, "loss": "linear", "random_state": 42,
        }),
        "mlp": (MLPRegressor, {
            "max_iter": 200, "batch_size": 128, "learning_rate_init": 0.003,
            "early_stopping": False, "shuffle": False, "random_state": 42,
        }),
        "linear": (LinearRegression, {"n_jobs": 1}),
    }
    if spec.family not in constructors:
        raise ValueError(f"Familia de modelos desconocida: {spec.family}")
    constructor, defaults = constructors[spec.family]
    settings = {**defaults, **params}
    if spec.family in {"mlp", "hist_gradient_boosting"} and settings.get("early_stopping"):
        raise ValueError("La parada temprana aleatoria no está permitida en la validación temporal.")
    if spec.family == "mlp" and "hidden_layer_sizes" in settings:
        settings["hidden_layer_sizes"] = tuple(settings["hidden_layer_sizes"])
    return constructor(**settings)


@dataclass
class FittedDirectModel:
    spec: ModelSpec
    pipeline: Pipeline
    numeric_features: list[str]
    excluded_features: dict[str, str]
    fit_warnings: list[str]
    n_train: int
    training_target_max: pd.Period
    cutoff: pd.Period
    horizon: int

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        """Predice precios sin acceder a target_price ni a variables de t+h."""
        if "horizon" in frame and not frame["horizon"].eq(self.horizon).all():
            raise ValueError("El horizonte del diseño difiere del horizonte del modelo.")
        if "current_price" not in frame:
            raise KeyError("La predicción requiere current_price.")
        prices = pd.to_numeric(frame["current_price"], errors="coerce").to_numpy(dtype=float)
        if not (np.isfinite(prices) & (prices > 0)).all():
            raise ValueError("La predicción requiere un precio positivo y finito en cada origen.")
        if frame.empty:
            return np.array([], dtype=float)
        columns = self.numeric_features + ["province_code"]
        design = frame.reindex(columns=columns).copy()
        design[self.numeric_features] = design[self.numeric_features].replace([np.inf, -np.inf], np.nan)
        with threadpool_limits(limits=1):
            growth = np.asarray(self.pipeline.predict(design), dtype=float)
        with np.errstate(over="ignore", invalid="ignore", under="ignore"):
            predictions = prices * np.exp(growth / 100)
        if not (np.isfinite(predictions) & (predictions > 0)).all():
            raise FloatingPointError(
                "El modelo ha producido un precio no finito o no positivo; no se recorta la predicción."
            )
        return predictions


def fit_direct(spec: ModelSpec, design: pd.DataFrame, cutoff: object) -> FittedDirectModel:
    """Ajusta usando únicamente etiquetas observables hasta cutoff inclusive.

    Medianas, indicadores de ausencia, escalas, codificación de provincias y
    escala del objetivo se estiman sólo con esas filas de entrenamiento.
    Las advertencias se conservan en fit_warnings para el informe de ajuste.
    """
    if spec.feature_set not in FEATURE_BLOCKS:
        raise ValueError(f"Conjunto de variables desconocido: {spec.feature_set}")
    required = {"quarter", "target_quarter", "target_log_growth", "target_price", "current_price", "province_code", "horizon"}
    missing = required.difference(design.columns)
    if missing:
        raise KeyError(f"Faltan columnas del diseño: {sorted(missing)}")
    cutoff_period = pd.Period(cutoff, freq="Q")
    target_quarters = pd.PeriodIndex(design["target_quarter"], freq="Q")
    origin_quarters = pd.PeriodIndex(design["quarter"], freq="Q")
    valid = (
        (target_quarters <= cutoff_period)
        & (origin_quarters < target_quarters)
        & np.isfinite(design["current_price"]) & design["current_price"].gt(0)
        & np.isfinite(design["target_price"]) & design["target_price"].gt(0)
        & np.isfinite(design["target_log_growth"])
    )
    train = design.loc[valid].copy()
    if train.empty:
        raise ValueError("No hay etiquetas observables y válidas para ajustar en este origen.")
    if design["horizon"].nunique() != 1:
        raise ValueError("Un modelo directo debe ajustarse para un único horizonte.")
    horizon = int(design["horizon"].iloc[0])
    requested = FEATURE_BLOCKS[spec.feature_set]
    excluded: dict[str, str] = {}
    numeric_features: list[str] = []
    for column in requested:
        if column not in train:
            excluded[column] = "missing_column"
            continue
        train[column] = pd.to_numeric(train[column], errors="coerce").replace([np.inf, -np.inf], np.nan)
        if train[column].isna().all():
            excluded[column] = "empty_in_training"
        else:
            numeric_features.append(column)
    if not numeric_features:
        raise ValueError("No hay predictores numéricos observados en el entrenamiento.")
    numeric = Pipeline([
        ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
        ("scaler", StandardScaler()),
    ])
    preprocessing = ColumnTransformer([
        ("numeric", numeric, numeric_features),
        ("province", OneHotEncoder(handle_unknown="ignore", sparse_output=False), ["province_code"]),
    ], sparse_threshold=0)
    model = TransformedTargetRegressor(regressor=_make_estimator(spec), transformer=StandardScaler())
    pipeline = Pipeline([("preprocess", preprocessing), ("model", model)])
    with warnings.catch_warnings(record=True) as caught, threadpool_limits(limits=1):
        warnings.simplefilter("always")
        pipeline.fit(train[numeric_features + ["province_code"]], train["target_log_growth"])
    recorded_warnings = list(dict.fromkeys(f"{item.category.__name__}: {item.message}" for item in caught))
    return FittedDirectModel(
        spec=spec, pipeline=pipeline, numeric_features=numeric_features,
        excluded_features=excluded, fit_warnings=recorded_warnings,
        n_train=len(train), training_target_max=pd.Period(train["target_quarter"].max(), freq="Q"),
        cutoff=cutoff_period, horizon=horizon,
    )
