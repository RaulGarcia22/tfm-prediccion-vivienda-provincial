"""Pruebas de integridad temporal para el panel y las matrices de modelado."""

from __future__ import annotations

import importlib.util
import json
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PANEL_PATH = ROOT / "data" / "processed" / "panel_provincial_trimestral.parquet"
MODEL_DIR = ROOT / "data" / "model"


def load_module(filename: str, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, ROOT / filename)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"No se pudo cargar {filename}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


class PanelIntegrityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.panel = pd.read_parquet(PANEL_PATH)
        cls.panel["quarter"] = pd.PeriodIndex(cls.panel["quarter"], freq="Q")
        cls.ordered = cls.panel.sort_values(["province_code", "quarter"])

    def test_primary_key_is_unique_and_complete(self) -> None:
        keys = ["province_code", "quarter"]
        self.assertFalse(self.panel[keys].isna().any().any())
        self.assertFalse(self.panel.duplicated(keys).any())

    def test_expected_geographical_and_temporal_scope(self) -> None:
        self.assertEqual(self.panel["province_code"].nunique(), 52)
        self.assertEqual(self.panel["quarter"].min(), pd.Period("1995Q1"))
        self.assertGreaterEqual(self.panel["quarter"].max(), pd.Period("2026Q1"))
        counts = self.panel.groupby("quarter")["province_code"].nunique()
        self.assertTrue(counts.eq(52).all())
        expected_quarters = pd.period_range(self.panel["quarter"].min(), self.panel["quarter"].max(), freq="Q")
        for _, group in self.panel.groupby("province_code"):
            self.assertEqual(sorted(group["quarter"].tolist()), expected_quarters.tolist())

    def test_missing_target_observations_are_preserved(self) -> None:
        source = pd.read_parquet(ROOT / "data" / "processed" / "components" / "target.parquet")
        observed = self.panel["valor_tasado_m2"].notna().sum()
        self.assertEqual(observed, source["valor_tasado_m2"].notna().sum())
        self.assertGreater(self.panel["valor_tasado_m2"].isna().sum(), 0)

    def test_no_infinite_numeric_values(self) -> None:
        numeric = self.panel.select_dtypes(include=np.number)
        # Los tipos anulables de pandas producen una matriz object; convertir
        # explícitamente permite comprobar también columnas Float64 e Int64.
        values = numeric.to_numpy(dtype="float64", na_value=np.nan)
        self.assertFalse(np.isinf(values).any())

    def test_price_lags_are_strictly_backward_looking(self) -> None:
        lookup = self.ordered.set_index(["province_code", "quarter"])["valor_tasado_m2"]
        for lag in (1, 2, 4):
            keys = pd.MultiIndex.from_arrays([self.ordered["province_code"], self.ordered["quarter"] - lag])
            expected = pd.Series(lookup.reindex(keys).to_numpy(), index=self.ordered.index)
            actual = self.ordered[f"valor_tasado_m2_lag{lag}"]
            pd.testing.assert_series_equal(actual, expected, check_names=False)

    def test_targets_are_exact_forward_shifts(self) -> None:
        lookup = self.ordered.set_index(["province_code", "quarter"])["valor_tasado_m2"]
        for horizon in (1, 4, 8, 12):
            keys = pd.MultiIndex.from_arrays([self.ordered["province_code"], self.ordered["quarter"] + horizon])
            expected = pd.Series(lookup.reindex(keys).to_numpy(), index=self.ordered.index)
            actual = self.ordered[f"target_m2_t{horizon}"]
            pd.testing.assert_series_equal(actual, expected, check_names=False)

    def test_quarterly_price_changes_use_only_current_and_past_values(self) -> None:
        grouped = self.ordered.groupby("province_code")["valor_tasado_m2"]
        expected_qoq = grouped.pct_change(1, fill_method=None) * 100
        expected_yoy = grouped.pct_change(4, fill_method=None) * 100
        pd.testing.assert_series_equal(
            self.ordered["valor_tasado_m2_qoq"], expected_qoq, check_names=False
        )
        pd.testing.assert_series_equal(
            self.ordered["valor_tasado_m2_yoy"], expected_yoy, check_names=False
        )


class ModelMatrixTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.matrices = {
            name: pd.read_parquet(MODEL_DIR / f"dataset_{name}.parquet")
            for name in ("base", "ampliado", "reciente")
        }

    def test_all_matrices_preserve_panel_key(self) -> None:
        panel = pd.read_parquet(PANEL_PATH)
        for name, matrix in self.matrices.items():
            with self.subTest(matrix=name):
                self.assertFalse(matrix.duplicated(["province_code", "quarter"]).any())
                self.assertEqual(len(matrix), len(panel))
                pd.testing.assert_frame_equal(matrix[["province_code", "quarter"]], panel[["province_code", "quarter"]])

    def test_temporal_split_boundaries(self) -> None:
        for name, matrix in self.matrices.items():
            with self.subTest(matrix=name):
                quarter = pd.PeriodIndex(matrix["quarter"], freq="Q")
                expected = np.where(
                    quarter <= pd.Period("2018Q4"),
                    "development",
                    np.where(quarter <= pd.Period("2022Q4"), "calibration", "evaluation"),
                )
                np.testing.assert_array_equal(matrix["split"].astype(str), expected)

    def test_target_columns_are_explicit_and_not_duplicated(self) -> None:
        expected = {"target_m2_t1", "target_m2_t4", "target_m2_t8", "target_m2_t12"}
        for name, matrix in self.matrices.items():
            with self.subTest(matrix=name):
                target_like = {column for column in matrix if column.startswith("target_")}
                self.assertEqual(target_like, expected)


class AnnualPublicationDelayTests(unittest.TestCase):
    def test_annual_component_is_delayed_eight_quarters(self) -> None:
        build = load_module("03_build_panel_skeleton.py", "build_panel")
        source = pd.DataFrame(
            {
                "province_code": ["01", "01"],
                "year": [2018, 2019],
                "annual_value": [10.0, 20.0],
            }
        )
        expanded = build.expand_annual_component(source, ["annual_value"], 8)
        observed = expanded.loc[expanded["annual_value"].notna()]
        self.assertEqual(observed["quarter"].min(), pd.Period("2020Q4"))
        value_at_first_release = observed.loc[
            observed["quarter"].eq(pd.Period("2020Q4")), "annual_value"
        ].iloc[0]
        self.assertEqual(value_at_first_release, 10.0)


class SourceAvailabilityAuditTests(unittest.TestCase):
    def test_manifest_matches_actual_panel_and_discloses_assumptions(self) -> None:
        manifest = json.loads((ROOT / "data" / "reports" / "data_build_manifest.json").read_text(encoding="utf-8"))
        panel = pd.read_parquet(PANEL_PATH)
        self.assertEqual(manifest["rows"], len(panel))
        self.assertEqual(manifest["missing_target_rows"], int(panel["valor_tasado_m2"].isna().sum()))
        self.assertEqual(manifest["quarterly_predictor_delay"], 1)
        self.assertEqual(manifest["annual_predictor_delay_after_reference_year_q4"], 8)
        self.assertFalse(manifest["historical_vintages_reconstructed"])
        self.assertTrue(all(len(item["sha256"]) == 64 for item in manifest["raw_snapshot_files"].values()))

    def test_every_primary_numeric_predictor_has_an_availability_rule(self) -> None:
        audit = pd.read_csv(ROOT / "data" / "reports" / "source_availability_audit.csv")
        expected = set()
        for path in (ROOT / "data" / "processed" / "components").glob("*.parquet"):
            if path.stem in set(audit["component"]):
                component = pd.read_parquet(path)
                expected.update(component.select_dtypes(include=np.number).columns)
        self.assertEqual(set(audit["variable"]), expected)
        self.assertTrue(audit["availability_rule"].notna().all())
        self.assertFalse(audit["historical_vintages_reconstructed"].any())


if __name__ == "__main__":
    unittest.main()
