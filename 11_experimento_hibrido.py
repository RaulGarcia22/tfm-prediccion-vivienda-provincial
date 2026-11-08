"""Prueba exploratoria de tendencia/Holt + modelos congelados, sin reselección reciente.

Reutiliza pronósticos rolling-origin ya ajustados: una mezcla de predicciones
no requiere reentrenar sus componentes. No cambia selección, bandas ni previsión 2028.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from forecasting.direct import ModelSpec
from forecasting.experiment import (ROOT, REPORTS, PANEL_PATH, HORIZONS, TUNING_TARGETS,
                                    SELECTION_CUTOFF, digest_json, write_json)
from forecasting.hybrid import BASES, WEIGHTS, KEYS, fit_weights, apply_weights
from forecasting.stages import (load_selection, validate_evaluation_manifest, file_sha256,
                                _expected_rows)
from forecasting.unified import read_panel, metric_table, pairwise_origin_bootstrap

OUT = ROOT / 'data/reports/hybrid_experiment'


def development_components(panel, selection):
    detail = pd.read_parquet(REPORTS / 'selection_predictions.parquet')
    chosen = pd.concat([
        detail.loc[detail.horizon.eq(h) & detail.candidate_id.eq(
            ModelSpec.from_dict(selection['selected_by_horizon'][str(h)]).key)]
        for h in HORIZONS
    ]).assign(role='selected')
    references = detail.loc[detail.family.isin(['persistence', *BASES])].copy()
    references['role'] = references.family
    result = pd.concat([chosen, references], ignore_index=True)
    # El parquet de selección no tiene manifiesto propio: contrastar su muestra
    # y sus etiquetas con el panel, además de la huella de selección vigente.
    expected = pd.concat([
        _expected_rows(panel, h, str(pd.Period(target) - h), target).assign(horizon=h)
        for h in HORIZONS for target in TUNING_TARGETS
    ]).set_index(KEYS).sort_index()
    for role, part in result.groupby('role'):
        part = part.set_index(KEYS).sort_index()
        if not part.index.equals(expected.index):
            raise ValueError(f'Muestra de desarrollo incompleta: {role}')
        for column in ['forecast_quarter', 'actual', 'current_price']:
            if not part[column].equals(expected[column]):
                raise ValueError(f'{column} de desarrollo incompatible con el panel.')
    for h, part in chosen.groupby('horizon'):
        mae = float((part.prediction - part.actual).abs().mean())
        if not np.isclose(mae, selection['selected_validation_MAE'][str(h)], rtol=1e-12):
            raise ValueError('Predicciones incompatibles con el MAE congelado.')
    return result


def main():
    import matplotlib
    matplotlib.use('Agg')
    OUT.mkdir(parents=True, exist_ok=True)
    panel = read_panel(PANEL_PATH)
    selection = load_selection(panel)
    protocol = {
        'experiment': 'convex_price_blend_v1',
        'status': 'exploratory_post_evaluation_design',
        'formula': '(1 - ml_weight) * baseline + ml_weight * original_selected',
        'bases': list(BASES), 'ml_weight_grid': list(WEIGHTS),
        'weight_scope': 'one global weight per base, same for every horizon and province',
        'objective': 'mean across horizons of MAE / persistence MAE',
        'tie_break': 'smaller ML weight, then base name',
        'selection_target_cutoff': SELECTION_CUTOFF,
        'development_targets': list(TUNING_TARGETS),
        'evaluation_status': 'retrospective, inspected before designing this extension',
        'no_interval_claim': True,
        'original_selection_id': selection['selection_id'],
    }
    write_json(OUT / 'protocol.json', protocol)
    development = development_components(panel, selection)
    grid, frozen = fit_weights(development)
    grid.to_csv(OUT / 'development_grid.csv', index=False)
    frozen.update(protocol=protocol,
                  development_predictions_sha256=file_sha256(REPORTS / 'selection_predictions.parquet'),
                  experiment_code_sha256=digest_json([
                      file_sha256(ROOT / 'forecasting/hybrid.py'), file_sha256(__file__)]))
    frozen['experiment_id'] = digest_json(frozen)
    # Persistir la decisión ANTES de cargar las predicciones de evaluación.
    write_json(OUT / 'frozen_weights.json', frozen)
    print('[FROZEN]', frozen['weights'], 'preferred:', frozen['preferred_base'], flush=True)
    metric_table(apply_weights(development, frozen), ['role', 'horizon']).to_csv(
        OUT / 'development_metrics.csv', index=False)

    validate_evaluation_manifest(panel, selection)
    evaluation = pd.read_parquet(REPORTS / 'evaluation_predictions.parquet')
    hybrids = apply_weights(evaluation, frozen)
    detail = pd.concat([evaluation, hybrids], ignore_index=True)
    detail.to_parquet(OUT / 'evaluation_predictions.parquet', index=False)
    metrics = metric_table(detail, ['role', 'horizon'])
    metrics.to_csv(OUT / 'evaluation_metrics.csv', index=False)
    metric_table(detail, ['role', 'origin', 'horizon']).to_csv(OUT / 'evaluation_origin_metrics.csv', index=False)
    last_common = panel.quarter.max() - max(HORIZONS)
    balanced = detail.loc[pd.PeriodIndex(detail.origin, freq='Q') <= last_common]
    metric_table(balanced, ['role', 'horizon']).to_csv(OUT / 'evaluation_balanced_metrics.csv', index=False)
    paired = []
    for base in BASES:
        role = f'hybrid_{base}'
        comparison = detail.loc[detail.role.isin([role, 'annual_trend', 'holt', 'persistence'])].copy()
        comparison['role'] = comparison.role.replace({role: 'selected'})
        paired.append(pairwise_origin_bootstrap(comparison, ['persistence', 'annual_trend', 'holt']).assign(hybrid=role))
    pd.concat(paired, ignore_index=True).to_csv(OUT / 'paired_uncertainty.csv', index=False)
    pivot = metrics.pivot(index='horizon', columns='role', values='MAE')
    scores = []
    for role in pivot:
        def wins(reference):
            # El orden de suma puede diferir unas ULP aunque el peso sea 0 o 1.
            # No contar esas diferencias de redondeo como mejoras predictivas.
            return int(((pivot[role] < pivot[reference]) &
                        ~np.isclose(pivot[role], pivot[reference], rtol=1e-10, atol=1e-10)).sum())
        scores.append({'role': role, 'mean_horizon_MAE': float(pivot[role].mean()),
                       'wins_vs_persistence': wins('persistence'),
                       'wins_vs_annual_trend': wins('annual_trend'),
                       'wins_vs_holt': wins('holt'),
                       'wins_vs_original_selected': wins('selected')})
    summary = pd.DataFrame(scores).sort_values('mean_horizon_MAE')
    summary.to_csv(OUT / 'summary.csv', index=False)
    write_json(OUT / 'manifest.json', {
        'status': 'complete', 'created_at': datetime.now(timezone.utc).isoformat(),
        'experiment_id': frozen['experiment_id'], 'original_selection_id': selection['selection_id'],
        'evaluation_manifest_sha256': file_sha256(REPORTS / 'evaluation_manifest.json'),
        'artifact_hashes': {p.name: file_sha256(p) for p in sorted(OUT.iterdir())
                            if p.is_file() and p.suffix in {'.json', '.csv', '.parquet'}
                            and p.name != 'manifest.json'},
    })
    from reporting.hybrid_report import write_hybrid_report
    write_hybrid_report()
    print(summary.to_string(index=False), flush=True)
    print('[REPORT] docs/experimento_hibrido.md', flush=True)


if __name__ == '__main__':
    main()
