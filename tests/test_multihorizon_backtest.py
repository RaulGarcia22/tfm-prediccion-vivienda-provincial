"""Pruebas del evaluador histórico a 1-12 trimestres."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from forecasting.multihorizon import (
    LastValueModel,
    backtest_provincial_series,
    metrics_by_band,
    metrics_by_horizon,
    run_multihorizon_backtest,
)


class AuditModel:
    name = "audit"

    def __init__(self) -> None:
        self.last_seen: pd.Period | None = None
        self.value: float | None = None

    def fit(self, history: pd.Series) -> "AuditModel":
        self.last_seen = history.index.max()
        self.value = float(history.iloc[-1])
        return self

    def forecast(self, steps: int) -> np.ndarray:
        assert self.value is not None
        return np.repeat(self.value, steps)


class ProvincialBacktestTests(unittest.TestCase):
    def setUp(self) -> None:
        quarters = pd.period_range("2020Q1", "2024Q4", freq="Q")
        self.series = pd.Series(np.arange(len(quarters), dtype=float) + 100, index=quarters)

    def test_model_only_sees_information_through_origin(self) -> None:
        model = AuditModel()
        result = backtest_provincial_series(
            self.series, model=model, origin="2022Q1", horizon=12
        )
        self.assertEqual(model.last_seen, pd.Period("2022Q1"))
        self.assertEqual(len(result), 11)
        self.assertEqual(result["horizon"].min(), 1)
        self.assertEqual(result["horizon"].max(), 11)

    def test_predictions_are_generated_without_real_intermediate_values(self) -> None:
        result = backtest_provincial_series(
            self.series, model=LastValueModel(), origin="2021Q4", horizon=4
        )
        expected = self.series.loc[pd.Period("2021Q4")]
        np.testing.assert_array_equal(result["prediction"], np.repeat(expected, 4))

    def test_origin_requires_an_observed_value(self) -> None:
        with self.assertRaisesRegex(ValueError, "origen"):
            backtest_provincial_series(
                self.series, model=LastValueModel(), origin="2019Q4", horizon=4
            )


class PanelBacktestTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        quarters = pd.period_range("2020Q1", "2024Q4", freq="Q")
        rows = []
        for code, province, offset in [("01", "Uno", 0), ("02", "Dos", 50)]:
            for index, quarter in enumerate(quarters):
                rows.append({
                    "province_code": code,
                    "province": province,
                    "quarter": str(quarter),
                    "valor_tasado_m2": 100 + offset + index,
                })
        cls.panel = pd.DataFrame(rows)

    def test_panel_results_preserve_province_origin_and_horizon(self) -> None:
        result = run_multihorizon_backtest(
            self.panel,
            model_factory=LastValueModel,
            origins=["2021Q4", "2022Q1"],
            horizon=12,
        )
        self.assertEqual(result["province_code"].nunique(), 2)
        self.assertEqual(result["origin"].nunique(), 2)
        self.assertEqual(result["horizon"].max(), 12)
        self.assertFalse(result[["prediction", "actual", "error"]].isna().any().any())

    def test_metric_tables_have_all_three_bands(self) -> None:
        result = run_multihorizon_backtest(
            self.panel,
            model_factory=LastValueModel,
            origins=["2021Q4"],
            horizon=12,
        )
        horizons = metrics_by_horizon(result)
        bands = metrics_by_band(result)
        self.assertEqual(set(horizons["horizon"]), set(range(1, 13)))
        self.assertEqual(
            set(bands["horizon_band"].astype(str)),
            {"corto_1_4", "medio_5_8", "largo_9_12"},
        )
        for column in ("MAE", "RMSE", "MAPE_pct", "mean_error"):
            self.assertTrue(np.isfinite(horizons[column]).all())


if __name__ == "__main__":
    unittest.main()
