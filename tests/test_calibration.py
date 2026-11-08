"""Calibración separada: rango finito, cobertura e integridad de las bandas."""
from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from forecasting.calibration import (
    apply_intervals, calibrate_intervals, interval_metrics, validate_calibration,
)


def calibration_detail():
    rows = []
    for horizon in (1, 4):
        for i in range(40):
            origin = pd.Period('2019Q1') + i // 20
            rows.append({'province_code': str(i % 20).zfill(2), 'origin': str(origin),
                         'forecast_quarter': str(origin + horizon), 'horizon': horizon,
                         'prediction': 1000.0, 'actual': 1000.0 * np.exp((i + 1) / 100)})
    return pd.DataFrame(rows)


class CalibrationTests(unittest.TestCase):
    def test_finite_sample_rank_is_an_observed_score(self):
        calibrated = calibrate_intervals(calibration_detail(), expected_horizons=(1, 4))
        part = calibrated.loc[calibrated.horizon.eq(1)].set_index('confidence')
        self.assertEqual(part.loc[.8, 'finite_sample_rank'], 33)
        self.assertEqual(part.loc[.95, 'finite_sample_rank'], 39)
        self.assertAlmostEqual(part.loc[.8, 'log_radius'], .33)
        self.assertAlmostEqual(part.loc[.95, 'log_radius'], .39)
        self.assertEqual(part.loc[.95, 'calibration_n'], 40)
        self.assertEqual(part.loc[.95, 'calibration_origins'], 2)

    def test_positive_nested_bounds_and_coverage(self):
        detail = calibration_detail()
        calibrated = calibrate_intervals(detail)
        bounded = apply_intervals(detail, calibrated)
        self.assertTrue(bounded.lower_95.le(bounded.lower_80).all())
        self.assertTrue(bounded.upper_95.ge(bounded.upper_80).all())
        self.assertTrue(bounded.lower_95.gt(0).all())
        metrics = interval_metrics(bounded)
        self.assertTrue(metrics.origins.eq(2).all())
        self.assertTrue(metrics.interpretation.eq('few_origins_descriptive_only').all())
        self.assertTrue(metrics.empirical_coverage.between(0, 1).all())
        self.assertTrue(metrics.interval_score.ge(metrics.mean_width_eur_m2).all())

    def test_rejects_duplicate_observations_and_mismatched_calendar(self):
        detail = calibration_detail()
        with self.assertRaisesRegex(ValueError, 'duplicadas'):
            calibrate_intervals(pd.concat([detail, detail.iloc[[0]]]))
        detail.loc[0, 'forecast_quarter'] = '2025Q1'
        with self.assertRaisesRegex(ValueError, 'Fecha objetivo'):
            calibrate_intervals(detail)

    def test_rejects_invalid_prices_and_small_samples(self):
        for value in (0, -1, np.nan, np.inf):
            detail = calibration_detail()
            detail.loc[0, 'actual'] = value
            with self.subTest(value=value), self.assertRaises(ValueError):
                calibrate_intervals(detail)
        with self.assertRaisesRegex(ValueError, 'insuficiente'):
            calibrate_intervals(calibration_detail().iloc[:5])

    def test_rejects_missing_horizons_and_confidences(self):
        detail = calibration_detail()
        with self.assertRaisesRegex(ValueError, 'horizonte'):
            calibrate_intervals(detail, expected_horizons=(1, 4, 8))
        calibrated = calibrate_intervals(detail)
        with self.assertRaisesRegex(ValueError, 'niveles'):
            apply_intervals(detail, calibrated.loc[calibrated.confidence.eq(.95)])
        with self.assertRaisesRegex(ValueError, 'horizonte'):
            apply_intervals(detail, calibrated.loc[calibrated.horizon.eq(1)])

    def test_rejects_invalid_and_non_nested_radii(self):
        for value in (-1, np.nan, np.inf):
            calibrated = calibrate_intervals(calibration_detail())
            calibrated.loc[0, 'log_radius'] = value
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, 'Radio'):
                validate_calibration(calibrated)
        calibrated = calibrate_intervals(calibration_detail())
        calibrated.loc[calibrated.confidence.eq(.8), 'log_radius'] = 1
        with self.assertRaisesRegex(ValueError, 'anidados'):
            validate_calibration(calibrated)

    def test_rejects_duplicate_calibration_and_bad_forecasts(self):
        detail = calibration_detail()
        calibrated = calibrate_intervals(detail)
        with self.assertRaisesRegex(ValueError, 'duplicados'):
            apply_intervals(detail, pd.concat([calibrated, calibrated.iloc[[0]]]))
        detail.loc[0, 'prediction'] = np.inf
        with self.assertRaisesRegex(ValueError, 'positivos y finitos'):
            apply_intervals(detail, calibrated)


if __name__ == '__main__':
    unittest.main()
