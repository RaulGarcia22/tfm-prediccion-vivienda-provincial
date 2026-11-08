"""Pruebas de los cuatro candidatos finales."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from forecasting.candidate_models import (
    DampedRecentTrendModel,
    GlobalHuberGrowthModel,
    HoltDampedModel,
    PersistenceModel,
)


class LocalCandidateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        quarters = pd.period_range("2010Q1", periods=40, freq="Q")
        cls.series = pd.Series(100 + 2 * np.arange(40), index=quarters)

    def test_all_local_candidates_return_twelve_positive_values(self) -> None:
        for model in (PersistenceModel(), DampedRecentTrendModel(), HoltDampedModel()):
            with self.subTest(model=model.name):
                forecast = model.fit(self.series).forecast(12)
                self.assertEqual(len(forecast), 12)
                self.assertTrue(np.isfinite(forecast).all())
                self.assertTrue((forecast > 0).all())

    def test_damped_trend_increments_decrease(self) -> None:
        forecast = DampedRecentTrendModel(damping=0.8).fit(self.series).forecast(12)
        log_increments = np.diff(np.r_[np.log(self.series.iloc[-1]), np.log(forecast)])
        self.assertTrue(np.all(np.diff(log_increments) < 0))


class GlobalHuberTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        rows = []
        quarters = pd.period_range("2010Q1", periods=48, freq="Q")
        for code, offset in [("01", 0), ("02", 50), ("03", 100)]:
            for index, quarter in enumerate(quarters):
                rows.append({
                    "province_code": code,
                    "province": f"P{code}",
                    "quarter": quarter,
                    "valor_tasado_m2": 500 + offset + 2 * index + 3 * np.sin(index / 3),
                })
        cls.panel = pd.DataFrame(rows)

    def test_training_targets_do_not_extend_beyond_origin(self) -> None:
        origin = pd.Period("2019Q4")
        train = GlobalHuberGrowthModel.training_frame(self.panel, origin)
        self.assertLessEqual((train["quarter"] + 1).max(), origin)

    def test_global_huber_forecasts_every_province_for_twelve_steps(self) -> None:
        model = GlobalHuberGrowthModel().fit(self.panel, "2019Q4")
        forecast = model.forecast(12)
        self.assertEqual(len(forecast), 3 * 12)
        self.assertEqual(forecast["province_code"].nunique(), 3)
        self.assertEqual(set(forecast["horizon"]), set(range(1, 13)))
        self.assertTrue(np.isfinite(forecast["prediction"]).all())


if __name__ == "__main__":
    unittest.main()
