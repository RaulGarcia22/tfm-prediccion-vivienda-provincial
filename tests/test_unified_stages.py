"""Pruebas de separación temporal, muestra común y cadena de manifiestos."""
from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

from forecasting import stages
from forecasting.direct import ModelSpec
from forecasting.experiment import HORIZONS, digest_json, protocol, write_json
from forecasting.unified import code_digest, frame_digest


def panel_fixture():
    return pd.DataFrame([
        {'province_code': code, 'quarter': quarter, 'valor_tasado_m2': 1000.0 + i}
        for code in ('01', '02')
        for i, quarter in enumerate(pd.period_range('2017Q1', '2026Q1', freq='Q'))
    ])


def selection_fixture(panel):
    selection = {'protocol': protocol(), 'implementation_hash': code_digest(),
                 'development_data_hash': frame_digest(panel.loc[panel.quarter.le(pd.Period('2018Q4'))]),
                 'selected_by_horizon': {str(h): ModelSpec('persistence', 'ar', {}).to_dict()
                                         for h in HORIZONS}}
    selection['selection_id'] = digest_json(selection)
    return selection


def task_fixture():
    panel = panel_fixture()
    spec = ModelSpec('ridge', 'ar', {'alpha': .1})
    detail = stages._expected_rows(panel, 12, '2022Q4', '2026Q1')
    detail['horizon'] = 12
    detail['candidate_id'] = spec.key
    detail['prediction'] = detail.current_price
    records = [{'origin': origin, 'training_target_max': origin, 'horizon': 12,
                'candidate_id': spec.key} for origin in sorted(detail.origin.unique())]
    return panel, spec, detail, records


class SelectionAndSampleTests(unittest.TestCase):
    def test_selected_baseline_is_fitted_once_and_reported_under_both_roles(self):
        panel = panel_fixture()
        selection = selection_fixture(panel)
        reference = ModelSpec('persistence', 'ar', {})

        def fake_task(data, spec_data, horizon, origins, cutoff, cache_root, fingerprint):
            detail = stages._expected_rows(data, horizon, '2022Q4', cutoff)
            detail['horizon'] = horizon
            detail['candidate_id'] = ModelSpec.from_dict(spec_data).key
            detail['prediction'] = detail.current_price
            records = [{'origin': origin, 'horizon': horizon, 'candidate_id': reference.key}
                       for origin in sorted(detail.origin.unique())]
            return detail, records

        with tempfile.TemporaryDirectory() as directory:
            with patch.object(stages, 'REPORTS', Path(directory)), \
                    patch.object(stages, 'HORIZONS', (1, 2)), \
                    patch.object(stages, 'baseline_specs', return_value=[reference]), \
                    patch.object(stages, 'evaluate_task', side_effect=fake_task) as evaluate:
                detail = stages.stage_predictions(panel, selection, 'evaluation', jobs=1)
                self.assertEqual(evaluate.call_count, 2)
                self.assertEqual(set(detail.role), {'selected', 'persistence'})
                selected = detail.loc[detail.role.eq('selected')].drop(columns='role').reset_index(drop=True)
                baseline = detail.loc[detail.role.eq('persistence')].drop(columns='role').reset_index(drop=True)
                pd.testing.assert_frame_equal(selected, baseline)

    def test_fit_failure_cannot_be_silently_dropped(self):
        panel = panel_fixture()
        selection = selection_fixture(panel)
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(stages, 'REPORTS', Path(directory)), \
                    patch.object(stages, 'HORIZONS', (1,)), \
                    patch.object(stages, 'evaluate_task', return_value=(pd.DataFrame(), [{'failure': 'failed fit'}])):
                with self.assertRaisesRegex(RuntimeError, 'failed fit'):
                    stages.stage_predictions(panel, selection, 'calibration', jobs=1)
                self.assertTrue((Path(directory) / 'calibration_failures.json').is_file())

    def test_selection_ignores_future_prices_but_rejects_development_changes(self):
        panel = panel_fixture()
        selection = selection_fixture(panel)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'selection.json'
            write_json(path, selection)
            with patch.object(stages, 'SELECTION_PATH', path):
                modified = panel.copy()
                modified.loc[modified.quarter.gt(pd.Period('2018Q4')), 'valor_tasado_m2'] *= 2
                self.assertEqual(stages.load_selection(modified)['selection_id'], selection['selection_id'])
                modified.loc[0, 'valor_tasado_m2'] *= 2
                with self.assertRaisesRegex(ValueError, 'desarrollo'):
                    stages.load_selection(modified)

    def test_selection_digest_and_horizon_schema_are_required(self):
        panel = panel_fixture()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'selection.json'
            selection = selection_fixture(panel)
            selection['selected_by_horizon'].pop('12')
            write_json(path, selection)
            with patch.object(stages, 'SELECTION_PATH', path):
                with self.assertRaisesRegex(ValueError, 'modificada'):
                    stages.load_selection(panel)
                selection.pop('selection_id')
                selection['selection_id'] = digest_json(selection)
                write_json(path, selection)
                with self.assertRaisesRegex(ValueError, 'horizontes'):
                    stages.load_selection(panel)

    def test_exact_common_sample_and_latest_known_training_label(self):
        panel, spec, detail, records = task_fixture()
        stages._validate_task_output(detail, records, panel, spec, 12, '2022Q4', '2026Q1')
        with self.assertRaisesRegex(ValueError, 'Muestras distintas'):
            stages._validate_task_output(detail.iloc[1:], records, panel, spec, 12, '2022Q4', '2026Q1')
        records[0]['training_target_max'] = '2023Q1'
        with self.assertRaisesRegex(ValueError, 'posteriores'):
            stages._validate_task_output(detail, records, panel, spec, 12, '2022Q4', '2026Q1')

    def test_corrupted_actuals_and_omitted_audits_are_rejected(self):
        panel, spec, detail, records = task_fixture()
        detail.loc[0, 'actual'] += 1
        with self.assertRaisesRegex(ValueError, 'actual'):
            stages._validate_task_output(detail, records, panel, spec, 12, '2022Q4', '2026Q1')
        panel, spec, detail, records = task_fixture()
        with self.assertRaisesRegex(ValueError, 'Auditoría'):
            stages._validate_task_output(detail, records[1:], panel, spec, 12, '2022Q4', '2026Q1')


class ArtifactChainTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        self.patch = patch.object(stages, 'REPORTS', self.directory)
        self.patch.start()
        self.addCleanup(self.patch.stop)
        self.panel = panel_fixture()
        self.selection = selection_fixture(self.panel)
        calibration = pd.DataFrame([
            {'horizon': h, 'confidence': c, 'log_radius': .1 if c == .8 else .2,
             'calibration_n': 40, 'calibration_origins': 5,
             'target_start': '2019Q1', 'target_end': '2022Q4'}
            for h in HORIZONS for c in (.8, .95)
        ])
        calibration.to_csv(self.directory / 'interval_calibration.csv', index=False)
        for stage in ('calibration', 'evaluation'):
            cutoff = pd.Period('2022Q4') if stage == 'calibration' else self.panel.quarter.max()
            artifacts = [f'{stage}_predictions.parquet', f'{stage}_fit_audit.json']
            if stage == 'calibration':
                artifacts += ['interval_calibration.csv']
            else:
                artifacts += ['evaluation_intervals.parquet', 'interval_coverage.csv', 'evaluation_metrics.csv']
            for name in artifacts:
                if not (self.directory / name).exists():
                    (self.directory / name).write_bytes(b'synthetic artifact')
            manifest = {'selection_id': self.selection['selection_id'], 'stage': stage,
                        'data_hash': frame_digest(self.panel.loc[self.panel.quarter.le(cutoff)]),
                        'implementation_hash': code_digest(), 'stage_code_hash': stages.stage_code_digest()}
            write_json(self.directory / f'{stage}_manifest.json', manifest)
            extra = {'interval_calibration_sha256': stages.file_sha256(self.directory / 'interval_calibration.csv')}
            if stage == 'evaluation':
                extra['calibration_manifest_sha256'] = stages.file_sha256(self.directory / 'calibration_manifest.json')
            stages._complete_manifest(stage, artifacts, **extra)

    def test_complete_valid_chain_loads(self):
        calibration = stages.load_calibration(self.panel, self.selection)
        self.assertEqual(len(calibration), 24)
        self.assertEqual(stages.validate_evaluation_manifest(self.panel, self.selection)['status'], 'complete')

    def test_modified_calibration_and_predictions_cannot_be_consumed(self):
        path = self.directory / 'interval_calibration.csv'
        path.write_bytes(path.read_bytes() + b'\n')
        with self.assertRaisesRegex(ValueError, 'Artefacto modificado'):
            stages.load_calibration(self.panel, self.selection)

    def test_recalibration_invalidates_existing_evaluation(self):
        path = self.directory / 'calibration_manifest.json'
        manifest = json.loads(path.read_text(encoding='utf-8'))
        manifest['extra_note'] = 'new calibration run'
        write_json(path, manifest)
        stages.load_calibration(self.panel, self.selection)
        with self.assertRaisesRegex(ValueError, 'otro manifiesto'):
            stages.validate_evaluation_manifest(self.panel, self.selection)

    def test_data_change_requires_the_relevant_stage_to_be_repeated(self):
        modified = self.panel.copy()
        modified.loc[modified.quarter.eq(pd.Period('2025Q4')), 'valor_tasado_m2'] += 1
        stages.load_calibration(modified, self.selection)
        with self.assertRaisesRegex(ValueError, 'evaluation'):
            stages.validate_evaluation_manifest(modified, self.selection)
        modified.loc[modified.quarter.eq(pd.Period('2021Q4')), 'valor_tasado_m2'] += 1
        with self.assertRaisesRegex(ValueError, 'calibration'):
            stages.load_calibration(modified, self.selection)


if __name__ == '__main__':
    unittest.main()
