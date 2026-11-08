"""Controles de fechas, muestras y congelación de las mezclas exploratorias."""
import unittest

import numpy as np
import pandas as pd

from forecasting.experiment import HORIZONS, TUNING_TARGETS
from forecasting.hybrid import aligned_predictions, blend, fit_weights, apply_weights


def fixture():
    return pd.DataFrame([
        {'province_code': province, 'origin': str(pd.Period(target) - h),
         'forecast_quarter': target, 'horizon': h, 'role': role,
         'prediction': prediction, 'actual': 100., 'current_price': 90.}
        for province in ['01', '02'] for h in HORIZONS for target in TUNING_TARGETS
        for role, prediction in [('selected', 80.), ('annual_trend', 120.),
                                 ('holt', 110.), ('persistence', 130.)]
    ])


class HybridTests(unittest.TestCase):
    def test_known_complementary_errors_choose_half_weight(self):
        grid, frozen = fit_weights(fixture())
        self.assertEqual(len(grid), 22)
        self.assertEqual(frozen['weights']['annual_trend'], .5)
        self.assertEqual(frozen['preferred_base'], 'annual_trend')
        hybrid = apply_weights(fixture(), frozen)
        self.assertTrue(hybrid.loc[hybrid.base.eq('annual_trend'), 'absolute_error'].eq(0).all())

    def test_future_labels_cannot_select_weights(self):
        future = fixture()
        future['forecast_quarter'] = (pd.PeriodIndex(future.forecast_quarter, freq='Q') + 40).astype(str)
        future['origin'] = (pd.PeriodIndex(future.origin, freq='Q') + 40).astype(str)
        with self.assertRaisesRegex(ValueError, 'posteriores'):
            fit_weights(future)

    def test_changing_evaluation_actuals_does_not_change_predictions(self):
        _, frozen = fit_weights(fixture())
        original = apply_weights(fixture(), frozen)
        changed = fixture()
        changed['actual'] = 1000.
        second = apply_weights(changed, frozen)
        np.testing.assert_array_equal(original.prediction, second.prediction)
        self.assertFalse(original.absolute_error.equals(second.absolute_error))

    def test_missing_rows_duplicates_and_corrupted_actuals_rejected(self):
        data = fixture()
        with self.assertRaisesRegex(ValueError, 'Muestras distintas'):
            aligned_predictions(data.iloc[1:])
        with self.assertRaisesRegex(ValueError, 'duplicadas'):
            aligned_predictions(pd.concat([data, data.iloc[:1]]))
        data.loc[0, 'actual'] += 1
        with self.assertRaisesRegex(ValueError, 'actual distinto'):
            aligned_predictions(data)

    def test_endpoints_are_allowed_and_bad_weights_rejected(self):
        np.testing.assert_array_equal(blend([1, 2], [3, 4], 0), [1, 2])
        np.testing.assert_array_equal(blend([1, 2], [3, 4], 1), [3, 4])
        for weight in [-.1, 1.1, np.nan]:
            with self.assertRaises(ValueError):
                blend([1], [2], weight)
        data = fixture()
        data.loc[data.role.eq('annual_trend'), 'prediction'] = 100.
        _, frozen = fit_weights(data)
        self.assertEqual(frozen['weights']['annual_trend'], 0)

    def test_missing_horizon_and_invalid_calendar_rejected(self):
        data = fixture()
        with self.assertRaisesRegex(ValueError, 'horizontes'):
            aligned_predictions(data.loc[data.horizon.ne(12)])
        data.loc[0, 'origin'] = '2000Q1'
        with self.assertRaisesRegex(ValueError, 'Calendario'):
            aligned_predictions(data)


if __name__ == '__main__':
    unittest.main()
