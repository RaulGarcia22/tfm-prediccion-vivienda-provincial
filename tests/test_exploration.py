"""Regresiones temporales del EDA: huecos y corte de selección."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("eda", ROOT / "04b_analisis_exploratorio.py")
eda = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = eda
SPEC.loader.exec_module(eda)


class ExploratoryCalendarTests(unittest.TestCase):
    def test_exact_horizon_does_not_jump_over_absent_quarters(self):
        data = pd.DataFrame({"province_code": ["01"] * 3,
                             "quarter": ["2020Q1", "2020Q3", "2021Q1"],
                             "valor_tasado_m2": [100., 125., 150.]})
        one = eda.exact_labels(data, 1, cutoff="2021Q1").set_index("quarter")
        self.assertTrue(one["forward_log_growth"].isna().all())
        four = eda.exact_labels(data, 4, cutoff="2021Q1").set_index("quarter")
        self.assertAlmostEqual(four.loc[pd.Period("2020Q1"), "forward_log_growth"], 100 * np.log(1.5))

    def test_no_target_after_development_cutoff_is_accepted(self):
        data = pd.DataFrame({"province_code": ["01", "01"],
                             "quarter": ["2018Q4", "2019Q1"],
                             "valor_tasado_m2": [100., 110.], "target_m2_t1": [110., 120.]})
        result = eda.exact_labels(data, 1)
        self.assertEqual(len(result), 1)
        self.assertTrue(result["forward_log_growth"].isna().all())

    def test_external_lag_preserves_gaps_and_does_not_shift_labels(self):
        data = pd.DataFrame({"province_code": ["01"] * 3,
                             "quarter": ["2018Q1", "2018Q3", "2018Q4"],
                             "valor_tasado_m2": [100., 110., 120.],
                             "euribor_12m": [1., 2., 3.],
                             "forward_log_growth": [7., 8., 9.]})
        result = eda.exact_feature_lags(data, ["euribor_12m"], 1)
        self.assertTrue(result["euribor_12m"].iloc[:2].isna().all())
        self.assertEqual(result["euribor_12m"].iloc[2], 2.)
        self.assertEqual(result["forward_log_growth"].tolist(), [7., 8., 9.])

    def test_future_observations_cannot_change_selection_reports(self):
        quarters = pd.period_range("2000Q1", "2026Q1", freq="Q")
        rng = np.random.default_rng(19)
        pieces = []
        for province in ("01", "02"):
            pieces.append(pd.DataFrame({"province_code": province, "quarter": quarters,
                "valor_tasado_m2": 1000 * np.exp(np.cumsum(rng.normal(.01, .02, len(quarters)))),
                "euribor_12m": rng.normal(size=len(quarters)),
                "target_m2_t4": 5e8}))
        data = pd.concat(pieces, ignore_index=True)
        data.loc[20:35, "euribor_12m"] = np.nan
        future_changed = data.copy()
        future = future_changed["quarter"] > pd.Period("2018Q4")
        future_changed.loc[future, ["valor_tasado_m2", "euribor_12m"]] = 1e10
        before = eda.selection_reports(data, horizons=(1, 4))
        after = eda.selection_reports(future_changed, horizons=(1, 4))
        for name in before:
            pd.testing.assert_frame_equal(before[name], after[name], obj=name)
        self.assertNotIn("target_m2_t4", before["correlaciones_crecimiento"]["variable"].unique())

    def test_calendar_audit_distinguishes_missing_rows_and_missing_prices(self):
        data = pd.DataFrame({"province_code": ["01"] * 4,
                             "quarter": ["2020Q1", "2020Q2", "2020Q4", "2021Q1"],
                             "valor_tasado_m2": [100., np.nan, 120., 130.]})
        report = eda.coverage_reports(data)["auditoria_calendario_provincial"].iloc[0]
        self.assertEqual(report["n_filas_calendario_ausentes"], 1)
        self.assertEqual(report["n_huecos_internos_precio"], 2)
        self.assertEqual(report["trimestres_sin_precio_internos"], "2020Q2;2020Q3")


if __name__ == "__main__":
    unittest.main()
