"""Bandas marginales por horizonte calibradas en un periodo separado.

Se aplica la corrección de rango de split conformal. La dependencia espacial
y temporal impide reclamar su garantía iid: se informa cobertura posterior.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

CONFIDENCES = (.8, .95)


def _positive_integer(values, name):
    numeric = pd.to_numeric(values, errors='coerce').to_numpy(dtype=float)
    if not (np.isfinite(numeric) & (numeric >= 1) & (numeric == np.floor(numeric))).all():
        raise ValueError(f'{name} debe contener enteros positivos.')
    return numeric.astype(int)


def validate_calibration(calibration, expected_horizons=None, confidences=CONFIDENCES):
    """Rechaza tablas incompletas, ambiguas o numéricamente inválidas."""
    required = {'horizon', 'confidence', 'log_radius', 'calibration_n', 'calibration_origins'}
    if calibration.empty or not required.issubset(calibration):
        raise ValueError('Calibración vacía o sin columnas obligatorias.')
    horizons = _positive_integer(calibration.horizon, 'horizon')
    n = _positive_integer(calibration.calibration_n, 'calibration_n')
    origins = _positive_integer(calibration.calibration_origins, 'calibration_origins')
    confidence = pd.to_numeric(calibration.confidence, errors='coerce').to_numpy(dtype=float)
    radius = pd.to_numeric(calibration.log_radius, errors='coerce').to_numpy(dtype=float)
    if not (np.isfinite(confidence) & (confidence > 0) & (confidence < 1)).all():
        raise ValueError('Confianza fuera de (0,1).')
    if set(confidence) != set(confidences):
        raise ValueError('Faltan niveles de confianza o hay niveles no autorizados.')
    if not (np.isfinite(radius) & (radius >= 0)).all():
        raise ValueError('Radio logarítmico negativo o no finito.')
    if (origins > n).any() or (np.ceil((n + 1) * confidence) > n).any():
        raise ValueError('Tamaño de calibración insuficiente o incoherente.')
    if calibration.duplicated(['horizon', 'confidence']).any():
        raise ValueError('Horizontes y niveles duplicados en calibración.')
    expected = set(horizons) if expected_horizons is None else set(expected_horizons)
    actual = set(zip(horizons, confidence))
    if actual != {(h, c) for h in expected for c in confidences}:
        raise ValueError('Falta calibración para algún horizonte o nivel.')
    for _, group in calibration.sort_values('confidence').groupby('horizon'):
        if (np.diff(group.log_radius.to_numpy(dtype=float)) < 0).any():
            raise ValueError('Los intervalos deben estar anidados por confianza.')
        if group.calibration_n.nunique() != 1 or group.calibration_origins.nunique() != 1:
            raise ValueError('Recuentos de calibración incoherentes entre niveles.')
    return calibration


def calibrate_intervals(detail: pd.DataFrame, confidences=CONFIDENCES,
                        expected_horizons=None) -> pd.DataFrame:
    required = {'province_code', 'horizon', 'actual', 'prediction', 'origin', 'forecast_quarter'}
    if not required.issubset(detail):
        raise ValueError(f'Faltan columnas: {required.difference(detail.columns)}')
    prices = detail[['actual', 'prediction']].to_numpy(dtype=float)
    if detail.empty or not (np.isfinite(prices) & (prices > 0)).all():
        raise ValueError('Calibración requiere precios positivos y finitos observados.')
    _positive_integer(detail.horizon, 'horizon')
    if detail[list(required)].isna().any().any():
        raise ValueError('Claves o datos de calibración ausentes.')
    if detail.duplicated(['province_code', 'origin', 'horizon']).any():
        raise ValueError('Observaciones de calibración duplicadas.')
    origins = pd.PeriodIndex(detail.origin, freq='Q')
    targets = pd.PeriodIndex(detail.forecast_quarter, freq='Q')
    if not (origins + detail.horizon.to_numpy() == targets).all():
        raise ValueError('Fecha objetivo incompatible con origen y horizonte.')
    if not confidences or len(set(confidences)) != len(confidences):
        raise ValueError('Niveles de confianza vacíos o duplicados.')
    if any(not np.isfinite(c) or not 0 < c < 1 for c in confidences):
        raise ValueError('Confianza fuera de (0,1).')
    rows = []
    for horizon, part in detail.groupby('horizon'):
        scores = np.abs(np.log(part.actual.to_numpy()) - np.log(part.prediction.to_numpy()))
        n = len(scores)
        ordered = np.sort(scores)
        for confidence in confidences:
            rank = int(np.ceil((n + 1) * confidence))
            if rank > n:
                raise ValueError('Calibración insuficiente para el nivel solicitado.')
            rows.append({'horizon': int(horizon), 'confidence': confidence,
                         'log_radius': float(ordered[rank - 1]), 'calibration_n': n,
                         'calibration_origins': part.origin.nunique(), 'finite_sample_rank': rank,
                         'target_start': str(part.forecast_quarter.min()),
                         'target_end': str(part.forecast_quarter.max()),
                         'method': 'pooled_absolute_log_error_finite_sample_rank'})
    return validate_calibration(pd.DataFrame(rows), expected_horizons, confidences)


def apply_intervals(forecast: pd.DataFrame, calibration: pd.DataFrame) -> pd.DataFrame:
    if forecast.empty or not {'horizon', 'prediction'}.issubset(forecast):
        raise ValueError('Previsión vacía o incompleta.')
    horizons = _positive_integer(forecast.horizon, 'horizon')
    validate_calibration(calibration)
    if not set(horizons).issubset(set(calibration.horizon)):
        raise ValueError('Falta calibración para algún horizonte.')
    prices = forecast.prediction.to_numpy(dtype=float)
    if not (np.isfinite(prices) & (prices > 0)).all():
        raise ValueError('Previsión debe contener precios positivos y finitos.')
    result = forecast.copy()
    for confidence, group in calibration.groupby('confidence'):
        radius = result.horizon.map(group.set_index('horizon').log_radius)
        suffix = int(round(confidence * 100))
        with np.errstate(over='ignore', under='ignore'):
            lower = prices * np.exp(-radius)
            upper = prices * np.exp(radius)
        if not (np.isfinite(lower) & (lower > 0) & np.isfinite(upper)).all():
            raise ValueError('El intervalo produce límites no positivos o no finitos.')
        result[f'lower_{suffix}'] = lower
        result[f'upper_{suffix}'] = upper
    return result


def interval_metrics(detail: pd.DataFrame, groups=('horizon',)) -> pd.DataFrame:
    if detail.empty:
        raise ValueError('No hay observaciones para medir cobertura.')
    required = set(groups) | {'actual', 'origin', 'lower_80', 'upper_80', 'lower_95', 'upper_95'}
    if not required.issubset(detail):
        raise ValueError('Faltan columnas para medir cobertura.')
    prices = detail[['actual', 'lower_80', 'upper_80', 'lower_95', 'upper_95']].to_numpy(dtype=float)
    if not (np.isfinite(prices) & (prices > 0)).all():
        raise ValueError('Precios y límites deben ser positivos y finitos.')
    if not (detail.lower_95.le(detail.lower_80) & detail.lower_80.le(detail.upper_80)
            & detail.upper_80.le(detail.upper_95)).all():
        raise ValueError('Límites de intervalo incoherentes.')
    rows = []
    for key, part in detail.groupby(list(groups), sort=True):
        keys = key if isinstance(key, tuple) else (key,)
        for confidence in CONFIDENCES:
            suffix = int(round(100 * confidence))
            lower, upper = part[f'lower_{suffix}'], part[f'upper_{suffix}']
            width = upper - lower
            alpha = 1 - confidence
            score = width + 2 / alpha * ((lower-part.actual).clip(lower=0) + (part.actual-upper).clip(lower=0))
            n_origins = part.origin.nunique()
            rows.append({**dict(zip(groups, keys)), 'nominal_coverage': confidence,
                         'empirical_coverage': part.actual.between(lower, upper).mean(),
                         'mean_width_eur_m2': width.mean(), 'interval_score': score.mean(),
                         'observations': len(part), 'origins': n_origins,
                         'interpretation': 'few_origins_descriptive_only' if n_origins < 8
                         else 'dependent_origins_no_iid_guarantee'})
    return pd.DataFrame(rows)

