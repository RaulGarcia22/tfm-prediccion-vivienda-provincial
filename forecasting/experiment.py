"""Protocolo único: selección, calibración, evaluación y despliegue.

Los límites se aplican a fechas objetivo, no solo a filas de origen. La
evaluación es retrospectiva y reutiliza una historia ya examinada por el autor.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PANEL_PATH = ROOT / 'data/processed/panel_provincial_trimestral.parquet'
REPORTS = ROOT / 'data/reports/unified'
SELECTION_PATH = REPORTS / 'selection.json'
HORIZONS = tuple(range(1, 13))
ANCHORS = (1, 4, 8, 12)
TUNING_TARGETS = ('2013Q4', '2015Q4', '2017Q4', '2018Q4')
SELECTION_CUTOFF = '2018Q4'
CALIBRATION_START = '2018Q4'
CALIBRATION_CUTOFF = '2022Q4'
EVALUATION_START = '2022Q4'
EVALUATION_TARGET_START = '2023Q1'
FORECAST_ORIGIN = '2026Q1'
FORECAST_END = '2028Q4'
SEED = 42
SHORTLIST_SIZE = 4


def protocol() -> dict:
    return {
        'version': 2, 'target': 'valor medio tasado de vivienda libre, EUR/m2',
        'horizons': list(HORIZONS), 'screening_horizons': list(ANCHORS),
        'tuning_target_quarters': list(TUNING_TARGETS),
        'tuning_origins_rule': 'target_quarter - horizon',
        'selection_target_cutoff': SELECTION_CUTOFF,
        'calibration_origin_start': CALIBRATION_START,
        'calibration_target_cutoff': CALIBRATION_CUTOFF,
        'evaluation_origin_start': EVALUATION_START,
        'evaluation_target_start': EVALUATION_TARGET_START,
        'forecast_origin': FORECAST_ORIGIN, 'forecast_end': FORECAST_END,
        'shortlist_size': SHORTLIST_SIZE, 'random_state': SEED,
        'selection_rule': 'MAE por horizonte; empate exacto por identificador estable',
        'screening_rule': 'media de MAE/MAE persistencia por horizonte, cuatro familias distintas',
        'baseline_rule': 'todas las referencias participan en la selección final',
        'refit_rule': 'en cada origen, entrenar solo con target_quarter <= origin',
        'interval_rule': 'cuantiles de error log absoluto por horizonte, calibración separada',
        'interval_limit': 'dependencia temporal/espacial: cobertura nominal sin garantía iid',
        'evaluation_status': 'retrospectiva; periodos ya consultados en desarrollo anterior',
        'publication_assumption': 'origen anclado al último valor tasado publicado; exógenas rezagadas según auditoría',
    }


def digest_json(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str) + '\n', encoding='utf-8')
