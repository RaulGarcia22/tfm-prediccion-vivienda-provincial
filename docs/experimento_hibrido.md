# Experimento exploratorio: combinación de series temporales y ML

## Objetivo y diseño

Se evalúa si combinar una referencia temporal con los modelos seleccionados
reduce el error. Se prueban dos mezclas convexas en **escala de precios**:

`predicción = (1 − peso_ML) × referencia + peso_ML × selección_original`

La referencia es tendencia interanual o Holt amortiguado. El segundo componente
es el modelo ya elegido para cada horizonte (Huber, Ridge, regresión lineal o
Huber recursivo); no es una nueva búsqueda entre todos los modelos. Tampoco es
un modelo entrenado sobre residuos. Es integración explícita mediante combinación
ponderada de pronósticos. Conserva las variables de sus componentes originales.

Se prueban once pesos entre 0 y 1, en pasos de 0,1, para cada referencia: 22
configuraciones, incluidas referencias puras. Se ajusta **un único peso global por
referencia**, común a provincias y horizontes, para limitar grados de libertad.
Se minimiza la media entre los doce horizontes del MAE dividido por el MAE de
persistencia. Los empates favorecen menor peso de ML y después el nombre de base.
Los extremos permiten concluir que mezclar no aporta si el desarrollo los elige.

Se usan únicamente las fechas objetivo 2013Q4, 2015Q4, 2017Q4 y 2018Q4 para
elegir pesos y referencia preferida. Los componentes son predicciones históricas
fuera del entrenamiento de cada origen. Se reutilizan los pronósticos guardados:
la combinación aritmética no necesita reajustar modelos. Se verifican muestra
común, etiquetas, selección y manifiestos. El peso se guarda antes de leer las
predicciones de evaluación. No se usa calibración para elegirlo.

**Límite metodológico:** el experimento se propone después de conocer la evaluación
original. Sus resultados recientes son exploratorios y retrospectivos; no constituyen
una prueba confirmatoria nueva. Además, los pesos y los modelos originales usan las
mismas fechas de desarrollo: sus errores de desarrollo tienen optimismo de selección,
no son una validación anidada. Solo hay cuatro fechas objetivo de selección por horizonte.

## Pesos congelados

- Holt: **20 % referencia + 80 % selección original**.
- Tendencia interanual: **0 % referencia + 100 % selección original**.

Un peso ML de 1 significa que la selección descarta la referencia y reproduce
exactamente el modelo original; no debe presentarse como un híbrido efectivo.
Un peso de 0 descarta ML y conserva solo la referencia.

La combinación preferida **en desarrollo** es Holt + selección. Esta elección
no se modifica para favorecer el resultado reciente. Los componentes siguen
reajustándose en cada origen conforme al experimento original; los pesos permanecen fijos.

Gráfico generado en local: `data/reports/hybrid_experiment/development_weights.png`.

## Resultados recientes

Objetivos desde 2023Q1 hasta 2026Q1, según horizonte. Cada método usa las mismas
provincias y orígenes dentro de cada horizonte. El promedio da el mismo peso a
cada horizonte; no es el error de una previsión nacional. Hay entre
2 y 13 orígenes por horizonte.

| Método | MAE medio por horizonte (€/m²) | Gana a tendencia (de 12) | Gana a selección original (de 12) |
| --- | ---: | ---: | ---: |
| Huber recursivo | 108.19 | 5 | 11 |
| Tendencia interanual | 109.25 | 0 | 12 |
| Holt | 127.98 | 0 | 11 |
| Tendencia amortiguada | 130.10 | 0 | 11 |
| Holt + selección | 151.04 | 0 | 11 |
| Tendencia + selección | 157.33 | 0 | 0 |
| Selección original | 157.33 | 0 | 0 |
| Persistencia | 182.68 | 0 | 0 |

Gráfico generado en local: `data/reports/hybrid_experiment/evaluation_comparison.png`.

La opción preferida en desarrollo alcanza **151.04 €/m²**
de MAE medio, frente a **109.25** de tendencia interanual
y **157.33** de la selección original. Mejora a la tendencia
en **0 de 12 horizontes**. A cuatro trimestres,
su MAE es **84.73 €/m²**, frente a
**56.72** de tendencia y **89.13**
de la selección original.

Mejorar al componente original no implica superar a la referencia sencilla ni
demuestra superioridad futura. Se informan las dos combinaciones, aunque una
obtenga mejores resultados recientes; no se elige retrospectivamente entre ellas.
`evaluation_balanced_metrics.csv` restringe todos los horizontes a los orígenes
comunes. `paired_uncertainty.csv` aporta intervalos descriptivos de diferencias de
error, remuestreando bloques de cuatro orígenes y manteniendo juntas las provincias;
con menos de ocho orígenes omite esos intervalos. No son garantías de significación
con datos independientes ni corrigen haber diseñado el experimento después de ver resultados.

No se han calibrado bandas específicas para las mezclas: las bandas originales
no pueden heredarse. Este experimento no demuestra mejoras en cobertura ni
cambia las previsiones finales publicadas hasta 2028.
Su incorporación como modelo final requeriría una nueva versión documentada
del protocolo, calibración propia y evaluación adicional sin usarla para ajustar.

## Reproducibilidad

Ejecutar desde la raíz, después del flujo original:

```powershell
.venv\Scripts\python.exe 11_experimento_hibrido.py
```

Los resultados están en `data/reports/hybrid_experiment/`. `protocol.json`
describe la búsqueda; `frozen_weights.json` fija decisiones y huellas;
`development_grid.csv` conserva toda la parrilla; `evaluation_predictions.parquet`
guarda los pronósticos por caso. `manifest.json` enlaza los resultados con la
selección y evaluación originales. El cuaderno 03 muestra los pesos de desarrollo
y el 04 la evaluación exploratoria. La extensión se ejecuta por separado del flujo
original para distinguir las conclusiones del estudio inicial de esta prueba posterior.
