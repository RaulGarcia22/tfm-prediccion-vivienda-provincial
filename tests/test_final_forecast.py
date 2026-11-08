"""Verifica el enlace selección -> ajuste final -> bandas y artefactos."""
from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from forecasting.direct import ModelSpec
from forecasting.experiment import PANEL_PATH, REPORTS, ROOT
from forecasting.final_forecast import file_sha256, forecast_selected, validate_final_forecast
from forecasting.stages import load_calibration, load_selection, validate_evaluation_manifest
from forecasting.unified import predict_at_origin, read_panel

BASENAME = "prevision_provincial_2026T2_2028T4"


def synthetic_panel() -> pd.DataFrame:
    rows = []
    for code in ("01", "02", "03"):
        for index, quarter in enumerate(pd.period_range("2000Q1", "2014Q4", freq="Q")):
            rows.append({"province_code": code, "province": "Provincia " + code,
                         "quarter": quarter,
                         "valor_tasado_m2": 1000 * np.exp(.002 * index + .03 * int(code) + .007 * np.sin(index)),
                         "viajeros_hoteles_por_1000_hab": 10 + index * .2 + int(code),
                         "pernoctaciones_hoteles_por_1000_hab": 25 + index * .4 + int(code)})
    return pd.DataFrame(rows)


def synthetic_selection(families=("persistence", "ridge")) -> dict:
    return {"selection_id": "selection-for-test", "selected_by_horizon": {
        str(h): ModelSpec(family, "tourism" if family == "ridge" else "ar",
                          {"alpha": 10.0} if family == "ridge" else {}).to_dict()
        for h, family in enumerate(families, 1)}}


def synthetic_calibration(horizons=(1, 2)) -> pd.DataFrame:
    return pd.DataFrame([{"horizon": h, "confidence": confidence, "log_radius": radius,
                          "calibration_n": 1000, "calibration_origins": 10}
                         for h in horizons for confidence, radius in ((.8, .05), (.95, .1))])


class FinalForecastRoutingTests(unittest.TestCase):
    def setUp(self):
        self.panel = synthetic_panel()
        self.selection = synthetic_selection()
        self.calibration = synthetic_calibration()
        self.origin, self.end = "2012Q4", "2013Q2"

    def test_routes_exact_selected_spec_per_horizon_and_only_observed_data(self):
        with patch("forecasting.final_forecast.predict_at_origin", wraps=predict_at_origin) as routed:
            forecast, audit = forecast_selected(self.panel, self.selection, self.calibration,
                                                self.origin, self.end)
        self.assertEqual(routed.call_count, 2)
        for call in routed.call_args_list:
            data, spec, horizon, origin = call.args
            self.assertLessEqual(data.quarter.max(), pd.Period(self.origin))
            self.assertEqual(spec, ModelSpec.from_dict(self.selection["selected_by_horizon"][str(horizon)]))
            self.assertIs(call.kwargs["require_actual"], False)
            part = forecast.loc[forecast.horizon.eq(horizon)]
            self.assertEqual(set(part.candidate_id), {spec.key})
        self.assertEqual(len(forecast), 6)
        self.assertEqual(audit[0]["prediction_strategy"], "local_reference")
        self.assertEqual(audit[1]["prediction_strategy"], "direct")
        self.assertIn("viajeros_hoteles_por_1000_hab", audit[1]["numeric_features"])
        self.assertLessEqual(pd.Period(audit[1]["training_target_max"]), pd.Period(self.origin))
        np.testing.assert_allclose(forecast.lower_80, forecast.prediction * np.exp(-.05))
        np.testing.assert_allclose(forecast.upper_95, forecast.prediction * np.exp(.1))

    def test_arbitrary_future_values_cannot_change_forecasts_or_fit(self):
        selection = synthetic_selection(("ridge", "ridge"))
        original, original_audit = forecast_selected(self.panel, selection, self.calibration,
                                                     self.origin, self.end)
        changed = self.panel.copy()
        future = changed.quarter.gt(pd.Period(self.origin))
        changed.loc[future, "valor_tasado_m2"] *= 1000
        changed.loc[future, "viajeros_hoteles_por_1000_hab"] = -1e10
        changed.loc[future, "pernoctaciones_hoteles_por_1000_hab"] = 1e12
        revised, revised_audit = forecast_selected(changed, selection, self.calibration,
                                                   self.origin, self.end)
        pd.testing.assert_frame_equal(original, revised)
        self.assertEqual(original_audit, revised_audit)

    def test_missing_origin_price_is_rejected_instead_of_dropping_province(self):
        self.panel.loc[self.panel.quarter.eq(pd.Period(self.origin)) & self.panel.province_code.eq("02"),
                       "valor_tasado_m2"] = np.nan
        with self.assertRaisesRegex(ValueError, "Todas las provincias"):
            forecast_selected(self.panel, self.selection, self.calibration, self.origin, self.end)

    def test_missing_horizon_selection_is_rejected(self):
        del self.selection["selected_by_horizon"]["2"]
        with self.assertRaisesRegex(ValueError, "selección de algún horizonte"):
            forecast_selected(self.panel, self.selection, self.calibration, self.origin, self.end)

    def test_target_horizon_mismatch_and_duplicate_cells_are_rejected(self):
        frame, _ = forecast_selected(self.panel, self.selection, self.calibration, self.origin, self.end)
        broken = frame.copy()
        broken.loc[0, "forecast_quarter"] = "2014Q4"
        with self.assertRaisesRegex(AssertionError, "objetivo"):
            validate_final_forecast(broken, ["01", "02", "03"], self.origin, self.end)
        with self.assertRaisesRegex(AssertionError, "cuadrícula"):
            validate_final_forecast(pd.concat([frame, frame.iloc[[0]]]), ["01", "02", "03"], self.origin, self.end)

    def test_wrong_selection_calibration_is_rejected_before_loading_bands(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            (directory / "calibration_manifest.json").write_text(json.dumps({
                "status": "complete", "stage": "calibration", "selection_id": "different-selection"
            }), encoding="utf-8")
            with patch("forecasting.stages.REPORTS", directory):
                with self.assertRaisesRegex(ValueError, "otra selección"):
                    load_calibration(self.panel, self.selection)


class FinalForecastArtifactTests(unittest.TestCase):
    def test_final_forecast_has_exact_grid_and_selected_models(self):
        panel = read_panel(PANEL_PATH)
        selection = load_selection(panel)
        calibration = load_calibration(panel, selection)
        validate_evaluation_manifest(panel, selection)
        forecast = pd.read_parquet(ROOT / "data/forecasts" / f"{BASENAME}.parquet")
        self.assertEqual(len(forecast), 52 * 11)
        self.assertEqual(forecast.province_code.nunique(), 52)
        self.assertEqual(set(forecast.selection_id), {selection["selection_id"]})
        internal = forecast.rename(columns={
            "trimestre": "forecast_quarter", "prediccion": "prediction",
            "limite_inferior_80": "lower_80", "limite_superior_80": "upper_80",
            "limite_inferior_95": "lower_95", "limite_superior_95": "upper_95"})
        validate_final_forecast(internal, list(panel.province_code.unique()))
        for horizon, part in forecast.groupby("horizon"):
            spec = ModelSpec.from_dict(selection["selected_by_horizon"][str(horizon)])
            self.assertEqual(set(part.candidate_id), {spec.key})
            self.assertEqual(set(part.family), {spec.family})
            self.assertEqual(set(part.feature_set), {spec.feature_set})
            for suffix in (80, 95):
                row = calibration.loc[calibration.horizon.eq(horizon) & calibration.confidence.eq(suffix / 100)].iloc[0]
                np.testing.assert_allclose(part[f"limite_inferior_{suffix}"], part.prediccion * np.exp(-row.log_radius))
                np.testing.assert_allclose(part[f"limite_superior_{suffix}"], part.prediccion * np.exp(row.log_radius))

    def test_config_and_fit_audit_link_actual_phase_artifacts(self):
        directory = ROOT / "data/forecasts"
        config = json.loads((directory / f"{BASENAME}_config.json").read_text(encoding="utf-8"))
        selection = json.loads((REPORTS / "selection.json").read_text(encoding="utf-8"))
        audit = json.loads((directory / f"{BASENAME}_fit_audit.json").read_text(encoding="utf-8"))
        self.assertEqual(config["selection_id"], selection["selection_id"])
        self.assertEqual(config["forecast_origin"], "2026Q1")
        self.assertEqual(config["forecast_end"], "2028Q4")
        self.assertEqual(len(audit), 11)
        self.assertEqual(config["nominal_confidences"], [.8, .95])
        for mapping in ("source_files", "output_files"):
            for relative, fingerprint in config[mapping].items():
                with self.subTest(artifact=relative):
                    self.assertEqual(file_sha256(ROOT / relative), fingerprint)
        for record in audit:
            horizon = str(record["horizon"])
            self.assertEqual(record["selection_id"], selection["selection_id"])
            self.assertEqual(record["specification"], selection["selected_by_horizon"][horizon])
            if record.get("training_target_max") is not None:
                self.assertLessEqual(pd.Period(record["training_target_max"]), pd.Period("2026Q1"))
            self.assertIn("warnings", record)
            self.assertIn("numeric_features", record)
            self.assertIn("excluded_features", record)
        self.assertEqual(len(list((directory / "figures").glob("*.png"))), 53)


if __name__ == "__main__":
    unittest.main()
