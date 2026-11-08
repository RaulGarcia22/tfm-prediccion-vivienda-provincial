"""Calendar fixtures exercise gaps, shuffled rows and publication boundaries."""
from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from forecasting.calendar import calendar_shift, complete_quarterly_grid

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("calendar_build", ROOT / "03_build_panel_skeleton.py")
build = importlib.util.module_from_spec(spec)
spec.loader.exec_module(build)


class CalendarTests(unittest.TestCase):
    def setUp(self):
        self.irregular = pd.DataFrame({
            "province_code": ["01", "01", "01", "02"],
            "quarter": ["2020Q4", "2020Q1", "2020Q2", "2020Q2"],
            "price": [140.0, 100.0, 110.0, 70.0],
        }, index=[9, 3, 7, 11])

    def test_grid_keeps_unobserved_price_missing(self):
        grid = complete_quarterly_grid(self.irregular)
        self.assertEqual(len(grid), 8)
        q3 = grid["quarter"].eq(pd.Period("2020Q3"))
        self.assertTrue(grid.loc[q3, "price"].isna().all())
        self.assertEqual(grid["price"].notna().sum(), 4)

    def test_lag_uses_calendar_and_preserves_original_index(self):
        actual = calendar_shift(self.irregular, "price", 1)
        self.assertEqual(actual.index.tolist(), [9, 3, 7, 11])
        self.assertTrue(np.isnan(actual.loc[9]))  # Q4 must not borrow Q2.
        self.assertEqual(actual.loc[7], 100.0)

    def test_future_labels_do_not_skip_a_missing_quarter(self):
        one = calendar_shift(self.irregular, "price", -1)
        two = calendar_shift(self.irregular, "price", -2)
        self.assertTrue(np.isnan(one.loc[7]))
        self.assertEqual(two.loc[7], 140.0)
        self.assertEqual(one.loc[3], 110.0)

    def test_duplicate_keys_fail(self):
        duplicate = pd.concat([self.irregular, self.irregular.iloc[:1]])
        with self.assertRaises(ValueError):
            complete_quarterly_grid(duplicate)
        with self.assertRaises(ValueError):
            calendar_shift(duplicate, "price", 1)

    def test_builder_features_preserve_gap(self):
        frame = self.irregular.rename(columns={"price": "valor_tasado_m2"})
        result = build.add_features(frame).set_index(["province_code", "quarter"])
        self.assertTrue(np.isnan(result.loc[("01", pd.Period("2020Q4")), "valor_tasado_m2_lag1"]))
        self.assertTrue(np.isnan(result.loc[("01", pd.Period("2020Q2")), "target_m2_t1"]))
        self.assertAlmostEqual(result.loc[("01", pd.Period("2020Q2")), "valor_tasado_m2_qoq"], 10.0)


class PublicationBoundaryTests(unittest.TestCase):
    def test_quarterly_predictor_enters_only_next_quarter(self):
        source = pd.DataFrame({"province_code": ["01"], "quarter": ["2020Q2"], "x": [50.0]})
        result = build.delay_quarterly_component(source)
        self.assertEqual(result.loc[0, "quarter"], pd.Period("2020Q3"))

    def test_population_release_conventions_are_not_double_lagged(self):
        annual = pd.DataFrame({"province_code": ["01", "01"], "year": [2018, 2019], "x": [100.0, 200.0]})
        quarterly = pd.DataFrame({"province_code": ["01"], "quarter": pd.PeriodIndex(["2021Q1"], freq="Q"), "x": [300.0]})
        with patch.object(build, "panel_end_quarter", return_value=pd.Period("2022Q1")):
            result = build.combine_population_releases(annual, quarterly, ["x"]).set_index("quarter")
        self.assertEqual(result.index.min(), pd.Period("2020Q4"))
        self.assertEqual(result.loc[pd.Period("2021Q1"), "x"], 100.0)
        self.assertEqual(result.loc[pd.Period("2021Q2"), "x"], 300.0)
        # A delayed release of annual 2019 must not overwrite newer quarterly 2021.
        self.assertEqual(result.loc[pd.Period("2021Q4"), "x"], 300.0)


if __name__ == "__main__":
    unittest.main()
