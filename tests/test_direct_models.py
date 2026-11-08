"""Auditoría temporal del diseño directo y de su ajuste por horizonte."""

from __future__ import annotations

import json
import unittest

import numpy as np
import pandas as pd

from forecasting.direct import (
    AR_FEATURES,
    FAMILY_LABELS,
    FEATURE_BLOCKS,
    ModelSpec,
    build_design,
    candidate_specs,
    fit_direct,
)


def make_panel() -> pd.DataFrame:
    rows = []
    for code, initial in [("01", 900), ("02", 1500)]:
        for index, quarter in enumerate(pd.period_range("2000Q1", "2011Q4", freq="Q")):
            rows.append({
                "province_code": code,
                "province": code,
                "quarter": quarter,
                "valor_tasado_m2": initial * np.exp(0.01 * index + 0.02 * np.sin(index / 2)),
                "euribor_12m": index / 10,
            })
    return pd.DataFrame(rows)


class DirectDesignTests(unittest.TestCase):
    def test_calendar_gap_does_not_shift_label_or_retard(self) -> None:
        panel = make_panel()
        missing = pd.Period("2005Q2")
        panel = panel.loc[~((panel["province_code"] == "01") & (panel["quarter"] == missing))]
        design = build_design(panel, 1).set_index(["province_code", "quarter"])
        before = design.loc[("01", pd.Period("2005Q1"))]
        after = design.loc[("01", pd.Period("2005Q3"))]
        self.assertTrue(np.isnan(before["target_price"]))
        self.assertTrue(np.isnan(after["growth_1q"]))
        self.assertTrue(np.isnan(after["volatility_4q"]))
        # El crecimiento anual puede conocerse aunque falte otro trimestre.
        self.assertTrue(np.isfinite(after["growth_4q"]))
        annual = build_design(panel, 4).set_index(["province_code", "quarter"])
        expected = panel.loc[(panel["province_code"] == "01") & (panel["quarter"] == "2006Q1"), "valor_tasado_m2"].iloc[0]
        self.assertEqual(annual.loc[("01", pd.Period("2005Q1")), "target_price"], expected)

    def test_legacy_shifted_columns_are_not_trusted(self) -> None:
        panel = make_panel()
        original = build_design(panel, 4)
        panel["growth_1q"] = 12345
        panel["valor_tasado_m2_lag1"] = -999
        panel["target_m2_t4"] = 0
        rebuilt = build_design(panel, 4)
        pd.testing.assert_frame_equal(original[AR_FEATURES], rebuilt[AR_FEATURES])

    def test_future_prices_and_exogenous_do_not_change_origin_predictors(self) -> None:
        panel = make_panel()
        cutoff = pd.Period("2007Q4")
        original = build_design(panel, 4)
        panel.loc[panel["quarter"] > cutoff, ["valor_tasado_m2", "euribor_12m"]] *= 20
        modified = build_design(panel, 4)
        before = original["quarter"] <= cutoff
        pd.testing.assert_frame_equal(
            original.loc[before, AR_FEATURES + ["euribor_12m"]],
            modified.loc[before, AR_FEATURES + ["euribor_12m"]],
        )

    def test_duplicate_keys_and_invalid_horizons_rejected(self) -> None:
        panel = make_panel()
        with self.assertRaisesRegex(ValueError, "duplicados"):
            build_design(pd.concat([panel, panel.iloc[:1]]), 4)
        for horizon in [0, -1, 1.5, True]:
            with self.assertRaises(ValueError):
                build_design(panel, horizon)

    def test_final_rows_without_targets_are_preserved(self) -> None:
        design = build_design(make_panel(), 12)
        self.assertEqual(len(design), len(make_panel()))
        final = design.loc[design["quarter"] == pd.Period("2011Q4")]
        self.assertTrue(final["target_price"].isna().all())
        self.assertTrue(final["target_quarter"].eq(pd.Period("2014Q4")).all())


class DirectFitTests(unittest.TestCase):
    def test_embargo_and_preprocessing_ignore_future_observations(self) -> None:
        panel = make_panel()
        cutoff = pd.Period("2007Q4")
        spec = ModelSpec("ridge", "credit", {"alpha": 10})
        design = build_design(panel, 4)
        first = fit_direct(spec, design, cutoff)
        panel.loc[panel["quarter"] > cutoff, ["valor_tasado_m2", "euribor_12m"]] *= 100
        second = fit_direct(spec, build_design(panel, 4), cutoff)
        self.assertEqual(first.n_train, 56)  # 2 provincias * 28 orígenes con etiqueta conocida.
        self.assertEqual(first.training_target_max, cutoff)
        origin = design.loc[design["quarter"] == cutoff]
        np.testing.assert_allclose(first.predict(origin), second.predict(origin), rtol=0, atol=0)
        scaler_one = first.pipeline.named_steps["preprocess"].named_transformers_["numeric"].named_steps["scaler"]
        scaler_two = second.pipeline.named_steps["preprocess"].named_transformers_["numeric"].named_steps["scaler"]
        np.testing.assert_array_equal(scaler_one.mean_, scaler_two.mean_)

    def test_prediction_does_not_require_or_use_future_labels(self) -> None:
        design = build_design(make_panel(), 4)
        fitted = fit_direct(ModelSpec("ridge", "ar", {"alpha": 1}), design, "2008Q4")
        origin = design.loc[design["quarter"] == pd.Period("2011Q4")]
        labels = ["target_price", "target_quarter", "target_log_growth"]
        np.testing.assert_array_equal(fitted.predict(origin), fitted.predict(origin.drop(columns=labels)))

    def test_missing_and_future_only_external_columns_are_audited(self) -> None:
        panel = make_panel()
        panel["importe_medio_hipoteca_eur"] = np.where(panel["quarter"] >= pd.Period("2009Q1"), 100000.0, np.nan)
        design = build_design(panel, 1)
        fitted = fit_direct(ModelSpec("ridge", "credit", {}), design, "2008Q4")
        self.assertEqual(fitted.excluded_features["importe_medio_hipoteca_eur"], "empty_in_training")
        self.assertEqual(fitted.excluded_features["compraventas_por_1000_hab"], "missing_column")
        self.assertIn("euribor_12m", fitted.numeric_features)

    def test_convergence_warnings_are_available_for_reporting(self) -> None:
        fitted = fit_direct(
            ModelSpec("mlp", "ar", {"max_iter": 1, "hidden_layer_sizes": [4]}),
            build_design(make_panel(), 4), "2008Q4",
        )
        self.assertTrue(any("ConvergenceWarning" in item for item in fitted.fit_warnings))

    def test_predict_rejects_wrong_horizon_and_invalid_current_price(self) -> None:
        design = build_design(make_panel(), 4)
        fitted = fit_direct(ModelSpec("ridge", "ar", {}), design, "2008Q4")
        origin = design.iloc[-2:].copy()
        origin["horizon"] = 8
        with self.assertRaisesRegex(ValueError, "horizonte"):
            fitted.predict(origin)
        origin["horizon"] = 4
        origin["current_price"] = 0
        with self.assertRaisesRegex(ValueError, "precio positivo"):
            fitted.predict(origin)

    def test_overflow_is_reported_instead_of_clipped(self) -> None:
        design = build_design(make_panel(), 4)
        fitted = fit_direct(ModelSpec("ridge", "ar", {}), design, "2008Q4")

        class OverflowPipeline:
            def predict(self, frame: pd.DataFrame) -> np.ndarray:
                return np.full(len(frame), 1e9)

        fitted.pipeline = OverflowPipeline()
        with self.assertRaisesRegex(FloatingPointError, "no se recorta"):
            fitted.predict(design.iloc[-2:])

    def test_search_specs_are_unique_serializable_and_cover_all_families(self) -> None:
        specs = candidate_specs()
        self.assertEqual(len(specs), 93)
        self.assertEqual(len({spec.key for spec in specs}), len(specs))
        self.assertEqual({spec.family for spec in specs}, set(FAMILY_LABELS))
        for spec in specs:
            reloaded = ModelSpec.from_dict(json.loads(json.dumps(spec.to_dict())))
            self.assertEqual(spec.key, reloaded.key)
        self.assertNotIn("saldo_migratorio_total", FEATURE_BLOCKS["all"])

    def test_every_family_can_fit_and_predict_with_missing_features(self) -> None:
        design = build_design(make_panel(), 4)
        for family in FAMILY_LABELS:
            spec = next(spec for spec in candidate_specs() if spec.family == family)
            with self.subTest(family=family):
                fitted = fit_direct(spec, design, "2008Q4")
                predictions = fitted.predict(design.iloc[-2:])
                self.assertEqual(len(predictions), 2)
                self.assertTrue(np.isfinite(predictions).all())
                self.assertTrue((predictions > 0).all())


if __name__ == "__main__":
    unittest.main()
