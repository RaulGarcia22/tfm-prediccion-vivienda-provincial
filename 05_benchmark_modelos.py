"""Selecciona conjuntamente modelos/variables para horizontes 1..12.

Cribado amplio en h1/4/8/12; cuatro familias finalistas y referencias en todos
los horizontes. Solo objetivos hasta 2018T4; calibración y evaluación separadas.
"""
from __future__ import annotations
import argparse
import json
import time
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from forecasting.direct import ModelSpec, candidate_specs
from forecasting.experiment import (
    ROOT, PANEL_PATH, REPORTS, SELECTION_PATH, HORIZONS, ANCHORS,
    TUNING_TARGETS, SELECTION_CUTOFF, SHORTLIST_SIZE, protocol, write_json, digest_json,
)
from forecasting.unified import (
    baseline_specs, read_panel, frame_digest, code_digest, evaluate_task, metric_table,
)


def origins_for_horizon(horizon: int) -> list[str]:
    return [str(pd.Period(target, freq='Q') - horizon) for target in TUNING_TARGETS]


def run_tasks(panel, specifications, horizons, jobs, fingerprint):
    tasks = [(spec, h) for spec in specifications for h in horizons]
    runner = Parallel(n_jobs=jobs, backend='loky', return_as='generator_unordered')
    results = runner(delayed(evaluate_task)(
        panel, spec.to_dict(), h, origins_for_horizon(h), SELECTION_CUTOFF,
        str(ROOT / 'data/cache/selection'), fingerprint,
    ) for spec, h in tasks)
    frames, audits = [], []
    started = time.monotonic()
    for index, (frame, audit) in enumerate(results, start=1):
        if not frame.empty:
            frames.append(frame)
        audits.extend(audit)
        if index % 8 == 0 or index == len(tasks):
            print(f'[SEARCH] {index}/{len(tasks)} tasks, {time.monotonic()-started:.0f}s', flush=True)
    if not frames:
        raise RuntimeError('No se pudo evaluar ningún candidato.')
    return pd.concat(frames, ignore_index=True), audits


def eligible_metrics(detail: pd.DataFrame, audit: list[dict]) -> pd.DataFrame:
    metrics = metric_table(detail, ['candidate_id', 'family', 'feature_set', 'horizon'])
    persistence_id = baseline_specs()[0].key
    reference = metrics.loc[metrics.candidate_id.eq(persistence_id)].set_index('horizon')
    failures = {(row['candidate_id'], row['horizon']) for row in audit if 'failure' in row}
    for index, row in metrics.iterrows():
        reference_rows = detail.loc[detail.candidate_id.eq(persistence_id) & detail.horizon.eq(row.horizon)]
        candidate_rows = detail.loc[detail.candidate_id.eq(row.candidate_id) & detail.horizon.eq(row.horizon)]
        keys = ['province_code', 'origin', 'horizon']
        reference_keys = set(map(tuple, reference_rows[keys].to_numpy()))
        candidate_keys = set(map(tuple, candidate_rows[keys].to_numpy()))
        metrics.loc[index, 'eligible'] = (
            row.origins == len(TUNING_TARGETS)
            and candidate_keys == reference_keys
            and (row.candidate_id, row.horizon) not in failures
        )
        metrics.loc[index, 'relative_MAE_persistence'] = row.MAE / reference.loc[row.horizon, 'MAE']
        metrics.loc[index, 'fit_warning_count'] = sum(len(rec.get('warnings', [])) for rec in audit
            if rec['candidate_id'] == row.candidate_id and rec['horizon'] == row.horizon)
    return metrics


def choose_shortlist(metrics: pd.DataFrame, specs: list[ModelSpec]) -> list[ModelSpec]:
    baseline_ids = {spec.key for spec in baseline_specs()}
    valid = metrics.loc[metrics.eligible.eq(True) & ~metrics.candidate_id.isin(baseline_ids)]
    ranks = valid.groupby(['candidate_id', 'family'], as_index=False).agg(
        relative_MAE=('relative_MAE_persistence', 'mean'), horizons=('horizon', 'nunique'))
    ranks = ranks.loc[ranks.horizons.eq(len(ANCHORS))].sort_values(['relative_MAE', 'candidate_id'])
    ranks.to_csv(REPORTS / 'screening_ranking.csv', index=False)
    chosen = ranks.drop_duplicates('family').head(SHORTLIST_SIZE).candidate_id.tolist()
    if len(chosen) < SHORTLIST_SIZE:
        raise RuntimeError('No hay suficientes familias completas para la preselección.')
    mapping = {spec.key: spec for spec in specs}
    return [mapping[key] for key in chosen]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--jobs', type=int, default=4)
    args = parser.parse_args()
    REPORTS.mkdir(parents=True, exist_ok=True)
    # Corte físico: ninguna búsqueda puede leer precios/objetivos posteriores.
    panel = read_panel(PANEL_PATH)
    development = panel.loc[panel.quarter.le(pd.Period(SELECTION_CUTOFF))].copy()
    specs = candidate_specs()
    baselines = baseline_specs()
    study = protocol()
    data_hash, implementation_hash = frame_digest(development), code_digest()
    fingerprint = digest_json([study, data_hash, implementation_hash])
    write_json(REPORTS / 'protocol.json', study)
    write_json(REPORTS / 'candidate_grid.json', [spec.to_dict() for spec in specs + baselines])
    print(f'[SCREEN] {len(specs)} configurations + {len(baselines)} baselines; data <= {SELECTION_CUTOFF}', flush=True)
    screened, screening_audit = run_tasks(development, specs + baselines, ANCHORS, args.jobs, fingerprint)
    screening_metrics = eligible_metrics(screened, screening_audit)
    screened.to_parquet(REPORTS / 'screening_predictions.parquet', index=False)
    screening_metrics.to_csv(REPORTS / 'screening_metrics.csv', index=False)
    write_json(REPORTS / 'screening_fit_audit.json', screening_audit)
    shortlist = choose_shortlist(screening_metrics, specs)
    print('[SHORTLIST]', [spec.to_dict() for spec in shortlist], flush=True)
    finalists = shortlist + baselines
    detail, audits = run_tasks(development, finalists, HORIZONS, args.jobs, fingerprint)
    metrics = eligible_metrics(detail, audits)
    metrics.to_csv(REPORTS / 'selection_metrics.csv', index=False)
    detail.to_parquet(REPORTS / 'selection_predictions.parquet', index=False)
    write_json(REPORTS / 'selection_fit_audit.json', audits)
    winners = metrics.loc[metrics.eligible.eq(True)].sort_values(['horizon', 'MAE', 'candidate_id']).drop_duplicates('horizon')
    if set(winners.horizon) != set(HORIZONS):
        raise RuntimeError('La selección no cubre los doce horizontes.')
    mapping = {spec.key: spec for spec in finalists}
    selection = {
        'protocol': study, 'development_data_hash': data_hash,
        'implementation_hash': implementation_hash, 'selection_fingerprint': fingerprint,
        'shortlist': [spec.to_dict() for spec in shortlist],
        'selected_by_horizon': {str(int(row.horizon)): mapping[row.candidate_id].to_dict()
                                for row in winners.itertuples()},
        'selected_validation_MAE': {str(int(row.horizon)): row.MAE for row in winners.itertuples()},
        'features_rule': 'se eliminan columnas vacías únicamente según entrenamiento de cada ajuste',
        'never_uses': 'calibration/evaluation outcomes to choose a candidate',
    }
    selection['selection_id'] = digest_json(selection)
    write_json(SELECTION_PATH, selection)
    winners.to_csv(REPORTS / 'selected_models.csv', index=False)
    # Aportación de bloques: comparar mejor ajuste de cada bloque solo en desarrollo.
    blocks = screening_metrics.loc[screening_metrics.eligible.eq(True)].groupby(
        ['family', 'feature_set', 'horizon'], as_index=False).MAE.min()
    ar = blocks.loc[blocks.feature_set.eq('ar'), ['family', 'horizon', 'MAE']].rename(columns={'MAE':'AR_MAE'})
    blocks = blocks.merge(ar, on=['family','horizon'], how='left')
    blocks['improvement_vs_AR_pct'] = 100 * (blocks.AR_MAE - blocks.MAE) / blocks.AR_MAE
    blocks.to_csv(REPORTS / 'feature_block_contribution.csv', index=False)
    print(winners[['horizon','family','feature_set','MAE']].to_string(index=False), flush=True)
    print(f'[FROZEN] {SELECTION_PATH}', flush=True)


if __name__ == '__main__':
    main()
