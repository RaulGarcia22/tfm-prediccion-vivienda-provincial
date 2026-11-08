"""Pruebas del contrato que une selección histórica y previsión final."""
from __future__ import annotations
import importlib.util
import json
from pathlib import Path
import unittest
import pandas as pd
from forecasting.direct import ModelSpec
from forecasting.experiment import (REPORTS, SELECTION_PATH, SELECTION_CUTOFF,
                                    CALIBRATION_CUTOFF, HORIZONS, TUNING_TARGETS)
from forecasting.stages import load_selection
from forecasting.unified import read_panel
from forecasting.experiment import PANEL_PATH


class SavedSelectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.selection = load_selection(read_panel(PANEL_PATH))

    def test_selection_covers_all_horizons_and_winners_exist_in_development(self):
        self.assertEqual(set(self.selection['selected_by_horizon']), {str(h) for h in HORIZONS})
        metrics = pd.read_csv(REPORTS / 'selection_metrics.csv')
        for horizon in HORIZONS:
            spec = ModelSpec.from_dict(self.selection['selected_by_horizon'][str(horizon)])
            eligible = metrics.loc[metrics.horizon.eq(horizon) & metrics.eligible.eq(True)]
            winner = eligible.sort_values(['MAE', 'candidate_id']).iloc[0]
            self.assertEqual(winner.candidate_id, spec.key)

    def test_selection_only_uses_prespecified_target_dates(self):
        detail = pd.read_parquet(REPORTS / 'selection_predictions.parquet')
        target = pd.PeriodIndex(detail.forecast_quarter, freq='Q')
        self.assertTrue((target <= pd.Period(SELECTION_CUTOFF)).all())
        self.assertEqual(set(detail.forecast_quarter), set(TUNING_TARGETS))
        origins = pd.PeriodIndex(detail.origin, freq='Q')
        self.assertTrue(all(t.ordinal-o.ordinal == h for t, o, h in zip(target, origins, detail.horizon)))

    def test_future_prices_cannot_change_validity_of_frozen_selection(self):
        panel = read_panel(PANEL_PATH)
        later = panel.quarter.gt(pd.Period(SELECTION_CUTOFF))
        panel.loc[later, 'valor_tasado_m2'] *= 100
        unchanged = load_selection(panel)
        self.assertEqual(unchanged['selection_id'], self.selection['selection_id'])

    def test_observed_training_labels_never_extend_after_fit_origin(self):
        audit = json.loads((REPORTS / 'selection_fit_audit.json').read_text(encoding='utf-8'))
        for record in audit:
            if 'training_target_max' in record:
                self.assertLessEqual(pd.Period(record['training_target_max']), pd.Period(record['origin']))


class RobustnessTests(unittest.TestCase):
    def test_leave_one_origin_diagnostic_preserves_input_and_exposes_instability(self):
        path = Path(__file__).resolve().parents[1] / '06_analisis_robustez.py'
        spec = importlib.util.spec_from_file_location('robustness_unified', path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        data = pd.DataFrame({'horizon':[1,1,1,1], 'origin':['a','b','a','b'],
                             'candidate_id':['chosen','chosen','other','other'],
                             'absolute_error':[1.,9.,4.,4.]})
        before = data.copy(deep=True)
        result = module.selection_sensitivity(data, {'1':'chosen'})
        pd.testing.assert_frame_equal(data, before)
        self.assertEqual(set(result.sensitivity_best_candidate), {'chosen','other'})
        self.assertTrue(result.role.eq('descriptive_only_no_reselection').all())


if __name__ == '__main__':
    unittest.main()
