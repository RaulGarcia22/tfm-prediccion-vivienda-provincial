# Protocolo experimental vigente

## Objeto y trazabilidad

Se predice el valor medio tasado nominal de vivienda libre provincial, en €/m², a horizontes de 1 a 12 trimestres. La previsión final usa los horizontes 1 a 11, porque 2026Q1 → 2028Q4 contiene once trimestres futuros. El horizonte 12 se conserva para comprobar comportamiento a tres años.

Identificador de selección: `626fb0f634f651e8641ebdf1d003dcbc7073b319a5d1c25371fa972810c27b6e`. Fuente ejecutable: `forecasting/experiment.py`; especificación congelada: `data/reports/unified/selection.json`.

## Fechas y usos permitidos

| Etapa | Orígenes | Fechas objetivo observadas | Uso |
| --- | --- | --- | --- |
| Desarrollo exploratorio | Hasta 2018Q4 | Hasta 2018Q4 | Asociaciones; no usa etiquetas posteriores |
| Selección | Para cada h: fecha objetivo menos h | 2013Q4, 2015Q4, 2017Q4, 2018Q4 | Elegir familia, variables e hiperparámetros |
| Calibración | Desde 2018Q4, limitados por t+h ≤ 2022Q4 | Posteriores a 2018Q4 y hasta 2022Q4 | Estimar radios de bandas |
| Evaluación | Desde 2022Q4; último origen depende de h | Desde 2023Q1 hasta 2026Q1 | Medir error, estabilidad y cobertura |
| Previsión final | 2026Q1 | 2026Q2–2028Q4, todavía desconocidos en el origen | Aplicar selección congelada |

La separación se aplica a **fechas objetivo**. En cada ajuste solo entran filas de entrenamiento con `target_quarter <= origin`. Para h=12, una fila originada en 2025 no aporta todavía una etiqueta al ajuste de 2026Q1. Los predictores de cada ejemplo son los disponibles en su propio origen, no los publicados cuando se conoce la etiqueta.

Las familias e hiperparámetros se fijan después de la selección y no cambian al avanzar el origen. Los coeficientes, medianas, escalas y categorías sí se vuelven a estimar utilizando el entrenamiento permitido en ese momento. Esto simula actualización del modelo sin repetir la selección.

## Selección conjunta

1. Se evalúan 93 configuraciones de 11 familias y cinco referencias en h=1,4,8,12. Ridge, Elastic Net y Huber prueban los siete conjuntos de variables; las restantes familias comparan la historia del precio y el conjunto completo.
2. Para cada candidato se calcula la media de `MAE candidato / MAE persistencia` en esos cuatro horizontes. Así un horizonte largo no domina automáticamente por su escala de error.
3. Se conservan cuatro candidatos pertenecientes a familias distintas. Es una búsqueda acotada: la mejor configuración de un horizonte intermedio puede no haber pasado ese cribado.
4. Los cuatro finalistas y las cinco referencias se evalúan en los doce horizontes. Gana el menor MAE de cada horizonte; un empate exacto se resuelve por identificador estable. Para ser elegible debe cubrir las mismas claves provincia-origen-horizonte que la referencia y no tener fallos de ajuste.
5. Se guardan especificaciones, variables, hiperparámetros, advertencias y huellas. Las etapas posteriores leen ese resultado; no escogen un nuevo modelo con los resultados recientes.

La cobertura puede diferir entre horizontes porque faltan valores de origen u objetivo; dentro de cada horizonte todos los candidatos elegibles deben compararse sobre las mismas observaciones.

## Modelos directos y referencias

El objetivo de los regresores directos es `100 × log(P[t+h]/P[t])`. Cada horizonte tiene su ajuste. La previsión se recupera con `P[t] × exp(crecimiento_estimado/100)`. Se usan precio pasado, identidad provincial, calendario y bloques externos de la fila de origen. No se precisan valores futuros de euríbor, turismo, renta ni población.

Las referencias son persistencia, tendencia interanual, tendencia reciente amortiguada, Holt amortiguado en logaritmos y Huber de crecimiento recursivo. La referencia Huber recursiva es diferente de Huber directo con turismo. Si Holt no dispone de ocho observaciones consecutivas recientes o la tendencia anual carece de su retardo, el comportamiento de respaldo está auditado; no se comprime el calendario eliminando ausencias.

La imputación por mediana, los indicadores de ausencia, el escalado y la codificación provincial se aprenden dentro de cada ajuste. Las columnas completamente vacías en el entrenamiento se excluyen según ese entrenamiento; el futuro no decide su inclusión.

## Métricas y comparación

MAE en €/m² es la pérdida principal. RMSE penaliza más los errores grandes; MAPE permite leer el error relativo; el sesgo es `predicción − observación`. El promedio provincial tiene el mismo peso por observación y no representa una estadística nacional ponderada por vivienda o población.

Se publican resultados por horizonte, provincia y origen, además de una tabla sobre orígenes comunes. El horizonte 12 dispone de menos evaluación reciente que el horizonte 1. Se evita interpretar la variación entre horizontes como un efecto puro de la distancia si cambian los orígenes.

Las diferencias de error frente a persistencia, tendencia anual y Holt se remuestrean en bloques móviles de cuatro orígenes, manteniendo juntas las provincias de cada origen. Si hay menos de ocho orígenes no se informa un intervalo bootstrap. Los intervalos disponibles son descriptivos; el solapamiento y la escasez de bloques limitan la inferencia.

## Bandas de predicción

Para cada horizonte se reúnen los errores de calibración `abs(log(actual/predicción))` de todas las provincias. Para un nivel c se usa el estadístico ordenado de rango `ceil((n+1) × c)`, con c=0,80 o 0,95. La banda es `[predicción × exp(−q), predicción × exp(q)]`.

La corrección de rango coincide con la usada en calibración dividida, pero aquí no se presupone intercambiabilidad: los casos están relacionados en el espacio y en el tiempo. Son bandas empíricas marginales por horizonte, con un radio relativo común a las provincias, y su cobertura se mide en el periodo posterior. No describen la probabilidad simultánea de que toda la trayectoria de un territorio quede dentro.

El radio calibrado se conserva en la previsión final para mantener la separación. Los coeficientes de los modelos se reajustan al origen final. Si la evaluación muestra infracobertura, se documenta; no se modifica el radio con esa misma evaluación.

## Robustez y límites de interpretación

El script 06 informa sensibilidad de la selección al retirar un origen, estabilidad territorial y temporal, y cambios entre horizontes de la trayectoria final. Es diagnóstico posterior; no altera `selection.json` ni ajusta una nueva combinación ganadora.


1. La variable objetivo es el valor medio tasado nominal de vivienda libre. No es un precio de cierre individual, un índice de calidad constante ni una valoración de un inmueble concreto; puede cambiar la composición de las viviendas tasadas.
2. Se usan las versiones descargadas de las series. No se han reconstruido todas sus publicaciones históricas y pueden contener revisiones que no se conocían en cada origen.
3. Los retrasos de disponibilidad son supuestos explícitos: un trimestre para las fuentes mensuales/trimestrales externas y ocho trimestres después del cuarto trimestre del año de referencia para las anuales. No sustituyen un calendario real de publicación. Los stocks trimestrales recientes de población prevalecen sobre los anuales retrasados cuando existen.
4. La selección utiliza solo cuatro fechas objetivo, todas en el cuarto trimestre. Son suficientes para ejecutar el protocolo, pero ofrecen evidencia limitada sobre regímenes y estacionalidad. No equivalen a cuatro ventanas largas de validación.
5. Las 93 configuraciones y la preselección también introducen riesgo de sobreajuste de validación. No se ha probado cada combinación posible ni se afirma optimalidad global. Las ablaciones comparan mejores configuraciones dentro de cada bloque, por lo que no identifican efectos causales de las variables.
6. Los periodos recientes ya habían sido inspeccionados en el desarrollo anterior. La separación actual evita usarlos para seleccionar automáticamente, pero no convierte la evaluación en un test ciego nuevo ni constituye una preregistración prospectiva.
7. Los errores están relacionados entre provincias y trimestres. La remuestra conjunta por bloques de orígenes es descriptiva; las bandas marginales por horizonte no son regiones de cobertura simultánea de toda la trayectoria.
8. Los horizontes largos disponen de pocos orígenes recientes. La tabla equilibrada utiliza los orígenes comunes a todos los horizontes y también tiene una muestra reducida. No hay evidencia suficiente para afirmaciones fuertes sobre estabilidad a tres años.
9. Cada horizonte puede seleccionar una familia distinta. La trayectoria puede contener cambios bruscos entre puntos; no se impone suavizado posterior porque alteraría el modelo evaluado. El informe de trayectoria los hace visibles.
10. Las bandas se calibran en un periodo anterior y los coeficientes se vuelven a estimar con los datos permitidos. Un cambio de dinámica puede causar sesgo e infracobertura. No se amplían después de mirar el test para aparentar una calibración correcta.
11. Los modelos comparten información provincial, pero no incorporan una matriz explícita de vecindad ni una estructura causal espacial. Las asociaciones con turismo, crédito u oferta son predictivas.
12. El origen final queda fijado en 2026Q1. La previsión es un ejercicio desde ese corte hasta 2028Q4; no afirma incorporar toda la información publicada a la fecha de lectura de este documento.
