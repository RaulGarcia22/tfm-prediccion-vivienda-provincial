# Resultados del estudio principal

Selección: `626fb0f634f651e8641ebdf1d003dcbc7073b319a5d1c25371fa972810c27b6e`. Las tablas proceden de
`data/reports/unified/` y de la previsión final en `data/forecasts/`.
Este informe técnico acompaña al código; la memoria académica se entrega aparte.

## Interpretación de la evaluación

La selección congelada obtiene menor MAE que persistencia en 12 de los 12 horizontes y que tendencia interanual en 0 de los 12. Su MAE medio entre horizontes es 157.33 €/m². El menor promedio retrospectivo entre los roles evaluados corresponde a Huber crecimiento recursivo (108.19 €/m²). Este promedio da el mismo peso a cada horizonte, aunque los horizontes tienen distintos orígenes; sobre los orígenes comunes el menor promedio corresponde a Huber crecimiento recursivo (84.32 €/m²).

La evaluación posterior no confirma que la selección sea la mejor opción en promedio. Se conserva la especificación congelada para que la previsión sea la aplicación del experimento realizado; reemplazarla por el ganador de esta evaluación exigiría una nueva selección y una evaluación futura. El pronóstico se presenta como resultado experimental, no como superioridad demostrada.

Las bandas nominales del 95 % cubren entre 31.7 % y 96.7 % de los casos por horizonte; 11 de los 12 horizontes quedan por debajo del 95 %. La menor cobertura aparece en h=12 (31.7 %). La infracobertura es material: estas bandas no proporcionan protección suficiente frente a la dinámica observada en el periodo reciente. Se publican con su cobertura empírica y no como una garantía del 95 % para 2028.

## Especificaciones seleccionadas

Los errores de esta tabla corresponden a desarrollo y no a la evaluación reciente.

| Horizonte | Modelo | Variables | MAE de selección (€/m²) | Orígenes |
| --- | --- | --- | --- | --- |
| 1 | Huber | Precio + turismo | 19.75 | 4 |
| 2 | Ridge | Precio + turismo | 24.65 | 4 |
| 3 | Ridge | Precio + turismo | 31.15 | 4 |
| 4 | Huber | Precio + turismo | 37.64 | 4 |
| 5 | Ridge | Precio + turismo | 44.86 | 4 |
| 6 | Huber crecimiento recursivo | Historia del precio | 55.32 | 4 |
| 7 | Huber | Precio + turismo | 66.84 | 4 |
| 8 | Regresión lineal | Historia del precio | 84.16 | 4 |
| 9 | Huber | Precio + turismo | 76.85 | 4 |
| 10 | Huber | Precio + turismo | 76.35 | 4 |
| 11 | Huber | Precio + turismo | 83.78 | 4 |
| 12 | Huber | Precio + turismo | 98.26 | 4 |

## MAE de evaluación por horizonte

Los métodos se comparan sobre los mismos casos dentro de cada horizonte.
El número de orígenes cambia entre horizontes.

| Horizonte | Holt log amortiguado | Huber crecimiento recursivo | Persistencia | Selección congelada | Tendencia interanual | Tendencia reciente amortiguada | Orígenes | Casos |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 23.80 | 21.45 | 34.42 | 26.31 | 20.60 | 21.60 | 13 | 676 |
| 2 | 36.58 | 32.17 | 60.01 | 46.55 | 29.92 | 34.40 | 12 | 624 |
| 3 | 50.60 | 44.14 | 85.82 | 67.67 | 41.63 | 50.32 | 11 | 572 |
| 4 | 68.45 | 59.46 | 112.74 | 89.13 | 56.72 | 69.21 | 10 | 520 |
| 5 | 88.85 | 77.62 | 139.33 | 117.91 | 75.01 | 90.22 | 9 | 468 |
| 6 | 108.11 | 93.11 | 164.86 | 93.11 | 90.78 | 111.32 | 8 | 416 |
| 7 | 126.81 | 107.48 | 189.88 | 168.49 | 107.13 | 131.92 | 7 | 364 |
| 8 | 151.23 | 127.28 | 219.45 | 181.15 | 129.01 | 155.33 | 6 | 312 |
| 9 | 178.81 | 149.38 | 250.32 | 230.86 | 150.81 | 181.54 | 5 | 260 |
| 10 | 204.85 | 169.81 | 279.80 | 261.68 | 176.16 | 208.34 | 4 | 208 |
| 11 | 232.14 | 193.84 | 310.34 | 288.13 | 205.24 | 237.56 | 3 | 156 |
| 12 | 265.54 | 222.49 | 345.19 | 317.01 | 228.04 | 269.41 | 2 | 104 |

## MAE sobre orígenes comunes a todos los horizontes

| Horizonte | Holt log amortiguado | Huber crecimiento recursivo | Persistencia | Selección congelada | Tendencia interanual | Tendencia reciente amortiguada |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 20.38 | 19.66 | 22.70 | 20.49 | 20.99 | 21.20 |
| 2 | 24.86 | 23.32 | 31.54 | 27.04 | 25.90 | 26.72 |
| 3 | 28.77 | 28.14 | 44.14 | 37.83 | 31.94 | 28.93 |
| 4 | 36.84 | 36.08 | 59.98 | 48.50 | 41.89 | 35.23 |
| 5 | 46.88 | 42.59 | 79.11 | 63.46 | 50.41 | 43.66 |
| 6 | 58.11 | 51.32 | 95.27 | 51.32 | 56.31 | 55.07 |
| 7 | 73.83 | 65.33 | 117.19 | 96.33 | 73.90 | 72.07 |
| 8 | 102.98 | 87.33 | 156.68 | 118.75 | 97.29 | 98.55 |
| 9 | 138.53 | 116.52 | 199.65 | 172.55 | 123.85 | 134.64 |
| 10 | 171.47 | 142.60 | 239.76 | 212.50 | 149.28 | 171.55 |
| 11 | 211.82 | 176.50 | 285.89 | 258.36 | 183.84 | 214.58 |
| 12 | 265.54 | 222.49 | 345.19 | 317.01 | 228.04 | 269.41 |

## Cobertura retrospectiva de las bandas

| Horizonte | Nominal (%) | Observada (%) | Anchura media (€/m²) | Orígenes |
| --- | --- | --- | --- | --- |
| 1 | 80.00 | 75.44 | 74.14 | 13 |
| 1 | 95.00 | 96.75 | 106.51 | 13 |
| 2 | 80.00 | 60.90 | 96.17 | 12 |
| 2 | 95.00 | 88.14 | 151.93 | 12 |
| 3 | 80.00 | 51.05 | 114.67 | 11 |
| 3 | 95.00 | 76.22 | 181.75 | 11 |
| 4 | 80.00 | 50.00 | 142.79 | 10 |
| 4 | 95.00 | 69.81 | 210.68 | 10 |
| 5 | 80.00 | 45.09 | 166.24 | 9 |
| 5 | 95.00 | 66.03 | 251.08 | 9 |
| 6 | 80.00 | 67.55 | 218.39 | 8 |
| 6 | 95.00 | 80.53 | 294.51 | 8 |
| 7 | 80.00 | 31.87 | 198.58 | 7 |
| 7 | 95.00 | 59.89 | 315.47 | 7 |
| 8 | 80.00 | 25.64 | 212.66 | 6 |
| 8 | 95.00 | 52.88 | 321.19 | 6 |
| 9 | 80.00 | 24.23 | 223.92 | 5 |
| 9 | 95.00 | 43.85 | 353.48 | 5 |
| 10 | 80.00 | 19.23 | 226.26 | 4 |
| 10 | 95.00 | 38.46 | 365.73 | 4 |
| 11 | 80.00 | 16.67 | 225.31 | 3 |
| 11 | 95.00 | 37.18 | 384.42 | 3 |
| 12 | 80.00 | 17.31 | 241.82 | 2 |
| 12 | 95.00 | 31.73 | 388.15 | 2 |

## Aportación de bloques en desarrollo

La mejora compara el mejor ajuste de cada bloque con el mejor ajuste AR de
su familia y horizonte. No identifica efectos causales de las variables.

| Familia | Variables | Mejora media frente a AR (%) | Horizontes |
| --- | --- | --- | --- |
| Elastic Net | Todos los bloques | -9.52 | 4 |
| Elastic Net | Historia del precio | 0.00 | 4 |
| Elastic Net | Precio + crédito y operaciones | -14.04 | 4 |
| Elastic Net | Precio + demografía | -10.71 | 4 |
| Elastic Net | Precio + economía | -10.06 | 4 |
| Elastic Net | Precio + oferta | -2.85 | 4 |
| Elastic Net | Precio + turismo | -0.01 | 4 |
| Huber | Todos los bloques | -15.19 | 4 |
| Huber | Historia del precio | 0.00 | 4 |
| Huber | Precio + crédito y operaciones | -20.03 | 4 |
| Huber | Precio + demografía | -12.78 | 4 |
| Huber | Precio + economía | -11.53 | 4 |
| Huber | Precio + oferta | -6.59 | 4 |
| Huber | Precio + turismo | 4.85 | 4 |
| Ridge | Todos los bloques | -14.83 | 4 |
| Ridge | Historia del precio | 0.00 | 4 |
| Ridge | Precio + crédito y operaciones | -21.00 | 4 |
| Ridge | Precio + demografía | -15.14 | 4 |
| Ridge | Precio + economía | -11.70 | 4 |
| Ridge | Precio + oferta | -6.62 | 4 |
| Ridge | Precio + turismo | 3.55 | 4 |

## Previsiones provinciales

El script 09 genera en local las previsiones desde 2026Q1 hasta 2028Q4,
manteniendo la especificación elegida para cada horizonte. Las tablas
individuales y las figuras se guardan en `data/forecasts/` y no se publican.

## Extensión híbrida

La combinación de referencias con los modelos seleccionados se documenta
por separado en [Experimento híbrido](experimento_hibrido.md).
Se trata de una prueba exploratoria posterior y no genera esta previsión.
