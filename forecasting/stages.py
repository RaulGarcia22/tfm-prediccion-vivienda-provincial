"""Etapas enlazadas por una selección congelada y manifiestos verificables."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from .direct import ModelSpec, candidate_specs
from .experiment import (ROOT, PANEL_PATH, REPORTS, SELECTION_PATH, SELECTION_CUTOFF,
                         CALIBRATION_START, CALIBRATION_CUTOFF, EVALUATION_START,
                         EVALUATION_TARGET_START, HORIZONS, write_json, digest_json, protocol)
from .unified import (read_panel, frame_digest, code_digest, baseline_specs, evaluate_task,
                      metric_table, pairwise_origin_bootstrap)
from .calibration import calibrate_intervals, apply_intervals, interval_metrics, validate_calibration


def file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def stage_code_digest():
    return hashlib.sha256(b''.join(
        (ROOT / 'forecasting' / name).read_bytes() for name in ('stages.py', 'calibration.py')
    )).hexdigest()


def load_selection(panel=None):
    selection = json.loads(SELECTION_PATH.read_text(encoding='utf-8'))
    content = dict(selection)
    recorded = content.pop('selection_id', None)
    if recorded is None or digest_json(content) != recorded:
        raise ValueError('Selección modificada después de congelarse.')
    if selection.get('protocol') != protocol():
        raise ValueError('La selección no corresponde al protocolo vigente.')
    chosen = selection.get('selected_by_horizon', {})
    if set(chosen) != {str(h) for h in HORIZONS}:
        raise ValueError('Selección incompleta: se necesitan todos los horizontes.')
    allowed = {spec.key for spec in candidate_specs() + baseline_specs()}
    for data in chosen.values():
        if not isinstance(data, dict) or set(data) != {'family', 'feature_set', 'params'}:
            raise ValueError('Especificación de modelo incompleta.')
        if not isinstance(data['params'], dict) or ModelSpec.from_dict(data).key not in allowed:
            raise ValueError('Modelo ajeno a la parrilla congelada.')
    if panel is not None:
        development = panel.loc[panel.quarter.le(pd.Period(SELECTION_CUTOFF))]
        if frame_digest(development) != selection['development_data_hash']:
            raise ValueError('Han cambiado los datos de desarrollo: volver a ejecutar 05.')
    if code_digest() != selection['implementation_hash']:
        raise ValueError('Ha cambiado la implementación: volver a ejecutar 05.')
    return selection


def _expected_rows(data, horizon, start, cutoff):
    """Muestra común independiente de las salidas de cada candidato."""
    current = data[['province_code', 'quarter', 'valor_tasado_m2']].rename(
        columns={'quarter': 'origin', 'valor_tasado_m2': 'current_price'})
    current = current.loc[current.origin.ge(pd.Period(start))
                          & (current.origin + horizon).le(pd.Period(cutoff))].copy()
    current['forecast_quarter'] = current.origin + horizon
    future = data[['province_code', 'quarter', 'valor_tasado_m2']].rename(
        columns={'quarter': 'forecast_quarter', 'valor_tasado_m2': 'actual'})
    expected = current.merge(future, on=['province_code', 'forecast_quarter'], validate='one_to_one')
    values = expected[['current_price', 'actual']].to_numpy(dtype=float)
    expected = expected.loc[(np.isfinite(values) & (values > 0)).all(axis=1)].copy()
    expected['origin'] = expected.origin.astype(str)
    expected['forecast_quarter'] = expected.forecast_quarter.astype(str)
    return expected.sort_values(['province_code', 'origin']).reset_index(drop=True)


def _validate_task_output(detail, records, data, spec, horizon, start, cutoff):
    expected = _expected_rows(data, horizon, start, cutoff)
    required = {'province_code', 'origin', 'forecast_quarter', 'horizon', 'candidate_id',
                'prediction', 'actual', 'current_price'}
    if detail.empty or not required.issubset(detail):
        raise ValueError('Candidato sin todas las observaciones evaluables.')
    if detail.duplicated(['province_code', 'origin', 'horizon']).any():
        raise ValueError('Predicciones duplicadas.')
    if not detail.horizon.eq(horizon).all() or not detail.candidate_id.eq(spec.key).all():
        raise ValueError('Predicciones de otro candidato u horizonte.')
    obtained = detail.sort_values(['province_code', 'origin']).reset_index(drop=True)
    columns = ['province_code', 'origin', 'forecast_quarter']
    if len(expected) != len(obtained) or not obtained[columns].equals(expected[columns]):
        raise ValueError('Muestras distintas: falta alguna provincia/origen o sobra información futura.')
    for column in ['actual', 'current_price']:
        if not np.array_equal(obtained[column].to_numpy(), expected[column].to_numpy()):
            raise ValueError(f'{column} no coincide con el panel autorizado.')
    prices = obtained.prediction.to_numpy(dtype=float)
    if not (np.isfinite(prices) & (prices > 0)).all():
        raise ValueError('Predicciones no positivas o no finitas.')
    audit_origins = [record.get('origin') for record in records]
    if len(set(audit_origins)) != len(audit_origins) or set(audit_origins) != set(expected.origin):
        raise ValueError('Auditoría de ajustes incompleta o duplicada.')
    baseline_families = {item.family for item in baseline_specs()}
    for record in records:
        if record.get('candidate_id') != spec.key or record.get('horizon') != horizon:
            raise ValueError('Auditoría de otro candidato u horizonte.')
        last_label = record.get('training_target_max')
        if spec.family not in baseline_families and last_label is None:
            raise ValueError('Falta fecha máxima de etiqueta de entrenamiento.')
        if last_label is not None and pd.Period(last_label) > pd.Period(record['origin']):
            raise ValueError('El entrenamiento utiliza etiquetas posteriores al origen.')


def _complete_manifest(stage, artifacts, **extra):
    path = REPORTS / f'{stage}_manifest.json'
    manifest = json.loads(path.read_text(encoding='utf-8'))
    manifest.update(extra)
    manifest['status'] = 'complete'
    manifest['artifact_hashes'] = {name: file_sha256(REPORTS / name) for name in artifacts}
    write_json(path, manifest)
    return manifest


def _validate_manifest(panel, selection, stage):
    path = REPORTS / f'{stage}_manifest.json'
    manifest = json.loads(path.read_text(encoding='utf-8'))
    cutoff = CALIBRATION_CUTOFF if stage == 'calibration' else str(panel.quarter.max())
    authorized = panel.loc[panel.quarter.le(pd.Period(cutoff))]
    if manifest.get('status') != 'complete' or manifest.get('stage') != stage:
        raise ValueError(f'La etapa {stage} no está completa.')
    if manifest.get('selection_id') != selection['selection_id']:
        raise ValueError(f'La etapa {stage} corresponde a otra selección.')
    if manifest.get('data_hash') != frame_digest(authorized):
        raise ValueError(f'Los datos de {stage} han cambiado.')
    if manifest.get('implementation_hash') != code_digest() or manifest.get('stage_code_hash') != stage_code_digest():
        raise ValueError(f'La implementación de {stage} ha cambiado: repetir la etapa.')
    required = {f'{stage}_predictions.parquet', f'{stage}_fit_audit.json'}
    required |= ({'interval_calibration.csv'} if stage == 'calibration'
                 else {'evaluation_intervals.parquet', 'interval_coverage.csv', 'evaluation_metrics.csv'})
    artifacts = manifest.get('artifact_hashes', {})
    if not required.issubset(artifacts):
        raise ValueError(f'Faltan artefactos en el manifiesto de {stage}.')
    for name, expected in artifacts.items():
        if Path(name).name != name:
            raise ValueError('Ruta de artefacto no válida.')
        if not (REPORTS / name).is_file() or file_sha256(REPORTS / name) != expected:
            raise ValueError(f'Artefacto modificado o ausente: {name}')
    return manifest


def load_calibration(panel, selection):
    """Carga sólo bandas completas vinculadas a selección, datos y código."""
    manifest = _validate_manifest(panel, selection, 'calibration')
    path = REPORTS / 'interval_calibration.csv'
    if manifest.get('interval_calibration_sha256') != file_sha256(path):
        raise ValueError('La tabla de calibración ha sido modificada.')
    calibration = pd.read_csv(path)
    validate_calibration(calibration, HORIZONS)
    for column in ('target_start', 'target_end'):
        if column not in calibration:
            raise ValueError('Faltan fechas de calibración.')
        dates = pd.PeriodIndex(calibration[column], freq='Q')
        if not ((dates > pd.Period(SELECTION_CUTOFF)) & (dates <= pd.Period(CALIBRATION_CUTOFF))).all():
            raise ValueError('Calibración fuera del periodo autorizado.')
    return calibration


def validate_evaluation_manifest(panel, selection):
    """Impide producir con evaluación/calibración ajenas o modificadas."""
    load_calibration(panel, selection)
    manifest = _validate_manifest(panel, selection, 'evaluation')
    if manifest.get('calibration_manifest_sha256') != file_sha256(REPORTS / 'calibration_manifest.json'):
        raise ValueError('La evaluación utilizó otro manifiesto de calibración.')
    if manifest.get('interval_calibration_sha256') != file_sha256(REPORTS / 'interval_calibration.csv'):
        raise ValueError('La evaluación utilizó otra calibración de intervalos.')
    return manifest


def stage_predictions(panel, selection, stage, jobs=4):
    if stage == 'calibration':
        start, cutoff = CALIBRATION_START, CALIBRATION_CUTOFF
    elif stage == 'evaluation':
        start, cutoff = EVALUATION_START, str(panel.quarter.max())
    else:
        raise ValueError(stage)
    REPORTS.mkdir(parents=True, exist_ok=True)
    data = panel.loc[panel.quarter.le(pd.Period(cutoff))].copy()
    fingerprint = digest_json([selection['selection_id'], frame_digest(data), stage, stage_code_digest()])
    origins = [str(p) for p in pd.period_range(start, pd.Period(cutoff)-1, freq='Q')]
    jobspec = []
    for horizon in HORIZONS:
        selected = ModelSpec.from_dict(selection['selected_by_horizon'][str(horizon)])
        jobspec.append(('selected', selected, horizon))
        if stage == 'evaluation':
            jobspec += [(spec.family, spec, horizon) for spec in baseline_specs()]
    # Un ganador puede ser también una referencia. Ajustarlo una sola vez
    # evita escribir simultáneamente la misma caché y garantiza igual salida.
    unique_jobs = {}
    for _, spec, horizon in jobspec:
        unique_jobs.setdefault((spec.key, horizon), (spec, horizon))
    tasks = iter(Parallel(n_jobs=jobs, backend='loky', return_as='generator')(
        delayed(evaluate_task)(data, spec.to_dict(), horizon, origins, cutoff,
                               str(ROOT / 'data/cache' / stage), fingerprint)
        for spec, horizon in unique_jobs.values()))
    completed = {}
    frames, audits = [], []
    for i, (role, spec, horizon) in enumerate(jobspec, 1):
        key = (spec.key, horizon)
        if key not in completed:
            completed[key] = next(tasks)
        detail, records = completed[key]
        failures = [record for record in records if 'failure' in record]
        if failures:
            write_json(REPORTS / f'{stage}_failures.json', failures)
            raise RuntimeError(f'Fallos en {stage}: {failures[0]}')
        _validate_task_output(detail, records, data, spec, horizon, start, cutoff)
        detail = detail.copy()
        detail['role'] = role
        frames.append(detail)
        audits += [dict(record, role=role) for record in records]
        print(f'[{stage.upper()}] {i}/{len(jobspec)} h{horizon} {role}', flush=True)
    result = pd.concat(frames, ignore_index=True)
    dates = pd.PeriodIndex(result.forecast_quarter, freq='Q')
    if stage == 'calibration' and not ((dates > pd.Period(SELECTION_CUTOFF)) & (dates <= pd.Period(CALIBRATION_CUTOFF))).all():
        raise AssertionError('Contaminación temporal en calibración.')
    if stage == 'evaluation' and not (dates >= pd.Period(EVALUATION_TARGET_START)).all():
        raise AssertionError('Contaminación temporal en evaluación.')
    result.to_parquet(REPORTS / f'{stage}_predictions.parquet', index=False)
    write_json(REPORTS / f'{stage}_fit_audit.json', audits)
    write_json(REPORTS / f'{stage}_manifest.json', {
        'selection_id': selection['selection_id'], 'data_hash': frame_digest(data),
        'stage': stage, 'status': 'predictions_complete',
        'implementation_hash': code_digest(), 'stage_code_hash': stage_code_digest(),
        'origin_min': result.origin.min(), 'origin_max': result.origin.max(),
        'target_min': result.forecast_quarter.min(), 'target_max': result.forecast_quarter.max(),
        'horizons': list(HORIZONS), 'common_sample_verified': True,
        'retrospective_evaluation': True,
    })
    return result


def run_calibration(jobs=4):
    panel = read_panel(PANEL_PATH)
    selection = load_selection(panel)
    detail = stage_predictions(panel, selection, 'calibration', jobs)
    calibration = calibrate_intervals(detail, expected_horizons=HORIZONS)
    calibration.to_csv(REPORTS / 'interval_calibration.csv', index=False)
    _complete_manifest('calibration', ['calibration_predictions.parquet', 'calibration_fit_audit.json',
                                      'interval_calibration.csv'],
                       interval_calibration_sha256=file_sha256(REPORTS / 'interval_calibration.csv'),
                       interval_assumption='marginal_pooled_dependent_errors_no_iid_guarantee')
    return calibration


def run_evaluation(jobs=4):
    panel = read_panel(PANEL_PATH)
    selection = load_selection(panel)
    calibration = load_calibration(panel, selection)
    detail = stage_predictions(panel, selection, 'evaluation', jobs)
    selected = apply_intervals(detail.loc[detail.role.eq('selected')], calibration)
    selected.to_parquet(REPORTS / 'evaluation_intervals.parquet', index=False)
    interval_metrics(selected).to_csv(REPORTS / 'interval_coverage.csv', index=False)
    interval_metrics(selected, ('province_code',)).to_csv(REPORTS / 'interval_coverage_province.csv', index=False)
    metric_table(detail, ['role', 'horizon']).to_csv(REPORTS / 'evaluation_metrics.csv', index=False)
    metric_table(detail, ['role', 'province_code']).to_csv(REPORTS / 'evaluation_province.csv', index=False)
    metric_table(detail, ['role', 'origin']).to_csv(REPORTS / 'evaluation_origin.csv', index=False)
    # Mismos orígenes para comparar horizontes además de la cobertura desigual.
    last_full_origin = panel.quarter.max() - max(HORIZONS)
    balanced = detail.loc[pd.PeriodIndex(detail.origin, freq='Q') <= last_full_origin]
    metric_table(balanced, ['role', 'horizon']).to_csv(REPORTS / 'evaluation_balanced_metrics.csv', index=False)
    pairwise_origin_bootstrap(detail, ['persistence', 'annual_trend', 'holt']).to_csv(
        REPORTS / 'evaluation_paired_uncertainty.csv', index=False)
    counts = detail.loc[detail.role.eq('selected')].groupby('horizon').origin.nunique()
    _complete_manifest('evaluation', [
        'evaluation_predictions.parquet', 'evaluation_fit_audit.json', 'evaluation_intervals.parquet',
        'interval_coverage.csv', 'interval_coverage_province.csv', 'evaluation_metrics.csv',
        'evaluation_province.csv', 'evaluation_origin.csv', 'evaluation_balanced_metrics.csv',
        'evaluation_paired_uncertainty.csv'],
        interval_calibration_sha256=file_sha256(REPORTS / 'interval_calibration.csv'),
        calibration_manifest_sha256=file_sha256(REPORTS / 'calibration_manifest.json'),
        origins_by_horizon={str(h): int(n) for h, n in counts.items()},
        horizons_with_fewer_than_8_origins=[int(h) for h, n in counts.items() if n < 8],
        balanced_origins=int(balanced.origin.nunique()),
        uncertainty_note='Horizontes largos tienen pocos orígenes: resultados descriptivos; no garantía iid.')
    return detail

