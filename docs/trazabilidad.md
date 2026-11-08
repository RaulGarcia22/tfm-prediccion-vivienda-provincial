# Trazabilidad de los resultados

Los identificadores siguientes vinculan los resultados con los datos, la
implementación y las especificaciones del estudio principal.

## Identificadores

- Selección: `626fb0f634f651e8641ebdf1d003dcbc7073b319a5d1c25371fa972810c27b6e`.
- Datos de desarrollo: `12a5101b8f612ddb6098fdf9f430839bc53557e41bdd4b7904e9191d56010240`.
- Implementación de selección: `39b45225e0118039f20c2ba1336288b94267df51a1ade2387dc450d83ca28b77`.
- Origen final de previsión: `2026Q1`.
- Último trimestre previsto: `2028Q4`.

## Configuraciones comparadas

| Familia | Configuraciones |
| --- | --- |
| AdaBoost | 4 |
| Elastic Net | 21 |
| Extra Trees | 4 |
| Gradient Boosting | 4 |
| HistGradientBoosting | 4 |
| Huber | 21 |
| K vecinos | 4 |
| Regresión lineal | 2 |
| MLP | 4 |
| Random Forest | 4 |
| Ridge | 21 |

Además participan cinco referencias temporales. Cada configuración se reestima
en varios orígenes y horizontes. Las auditorías conservan las advertencias de
ajuste y las exclusiones de variables realizadas dentro del entrenamiento.

## Alcance de cada componente

| Componente | Función | Relación con la previsión final |
| --- | --- | --- |
| Flujo principal, scripts 03–10 | Preparación, selección, calibración, evaluación, previsión y documentación | Genera las 572 previsiones publicadas |
| Exploración, script 04b | Cobertura y asociaciones; asociaciones restringidas a desarrollo | Apoya el análisis, sin seleccionar automáticamente modelos |
| Robustez, script 06 | Diagnósticos territoriales, temporales y de sensibilidad | Describe resultados; no modifica la selección |
| Extensión híbrida, script 11 | Combinación ponderada de referencias y selección original | Exploratoria y posterior; no sustituye la previsión final |
| Pruebas automatizadas | Verificación del calendario, disponibilidad e integridad | Comprueban restricciones del código, no superioridad predictiva |

El flujo principal es un experimento computacional retrospectivo de comparación
predictiva. No constituye una intervención causal ni una validación prospectiva.
La extensión híbrida tiene sus propios pesos, métricas y manifiesto en
`data/reports/hybrid_experiment/` y se ejecuta por separado.

## Inventario del estudio principal

- `calibration_fit_audit.json`
- `calibration_manifest.json`
- `calibration_predictions.parquet`
- `candidate_grid.json`
- `evaluation_balanced_metrics.csv`
- `evaluation_fit_audit.json`
- `evaluation_intervals.parquet`
- `evaluation_manifest.json`
- `evaluation_metrics.csv`
- `evaluation_origin.csv`
- `evaluation_paired_uncertainty.csv`
- `evaluation_predictions.parquet`
- `evaluation_province.csv`
- `feature_block_contribution.csv`
- `forecast_trajectory_diagnostics.csv`
- `interval_calibration.csv`
- `interval_coverage.csv`
- `interval_coverage_province.csv`
- `protocol.json`
- `robustness_leave_one_origin_out.csv`
- `robustness_selection_stability.csv`
- `robustness_temporal.csv`
- `robustness_territorial.csv`
- `screening_fit_audit.json`
- `screening_metrics.csv`
- `screening_predictions.parquet`
- `screening_ranking.csv`
- `selected_models.csv`
- `selection.json`
- `selection_fit_audit.json`
- `selection_metrics.csv`
- `selection_predictions.parquet`

Estos archivos se generan en local en `data/reports/unified/` y no se publican
en el repositorio. Los datos brutos y sus
huellas están descritos en `data/reports/data_build_manifest.json`; las reglas
de disponibilidad figuran en `data/reports/source_availability_audit.csv`.
La configuración final en `data/forecasts/` contiene las huellas de sus entradas
y salidas. Git conserva los bytes originales mediante `.gitattributes` para
mantener estas comprobaciones al clonar en distintos sistemas.
