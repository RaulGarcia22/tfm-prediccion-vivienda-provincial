# Predicción provincial del valor tasado de vivienda libre hasta 2028

Código y documentación del estudio de predicción trimestral del valor medio
tasado nominal de vivienda libre en España. La unidad de análisis es
provincia × trimestre: 52 territorios (50 provincias,
Ceuta y Melilla), 6500 filas entre 1995Q1 y
2026Q1, con 6411 valores
observados y 89 ausencias explícitas.
La variable se expresa en €/m² nominales.

## Estudio principal y extensión exploratoria

El estudio principal selecciona modelos y variables por horizonte con datos de
desarrollo hasta 2018Q4. La especificación generada en local en
`data/reports/unified/selection.json` se utiliza después en calibración,
evaluación y previsión final. Los modelos se reajustan sin repetir la selección.

El resultado es una **previsión de investigación con origen fijo en 2026Q1**:
572 predicciones para 52 territorios entre 2026Q2 y 2028Q4.
La evaluación es retrospectiva sobre periodos previamente inspeccionados.
No acredita una validación prospectiva ni el funcionamiento de un servicio
operativo de predicción.

La **extensión híbrida del script 11** es una prueba exploratoria posterior que
combina tendencia interanual o Holt con la selección original. Sus resultados
se generan en local en `data/reports/hybrid_experiment/`. No interviene
en la selección, las bandas ni las previsiones finales. Su diseño y resultados
están en [Experimento híbrido](docs/experimento_hibrido.md).

## Resultados principales

Huber con precio + turismo: 7 horizontes; Regresión lineal con historia del precio: 1 horizonte; Huber crecimiento recursivo con historia del precio: 1 horizonte; Ridge con precio + turismo: 3 horizontes.

La selección congelada obtiene menor MAE que persistencia en 12 de los 12 horizontes y que tendencia interanual en 0 de los 12. Su MAE medio entre horizontes es 157.33 €/m². El menor promedio retrospectivo entre los roles evaluados corresponde a Huber crecimiento recursivo (108.19 €/m²). Este promedio da el mismo peso a cada horizonte, aunque los horizontes tienen distintos orígenes; sobre los orígenes comunes el menor promedio corresponde a Huber crecimiento recursivo (84.32 €/m²).

La evaluación posterior no confirma que la selección sea la mejor opción en promedio. Se conserva la especificación congelada para que la previsión sea la aplicación del experimento realizado; reemplazarla por el ganador de esta evaluación exigiría una nueva selección y una evaluación futura. El pronóstico se presenta como resultado experimental, no como superioridad demostrada.

Las bandas nominales del 95 % cubren entre 31.7 % y 96.7 % de los casos por horizonte; 11 de los 12 horizontes quedan por debajo del 95 %. La menor cobertura aparece en h=12 (31.7 %). La infracobertura es material: estas bandas no proporcionan protección suficiente frente a la dinámica observada en el periodo reciente. Se publican con su cobertura empírica y no como una garantía del 95 % para 2028.

Las tablas están en [Resultados](docs/resultados.md). Los niveles del 80 % y
95 % de las bandas son nominales; deben interpretarse junto con su cobertura
retrospectiva observada.

## Fuentes oficiales de los datos

Los conjuntos de datos no se publican en este repositorio. La carpeta
`data/` completa está excluida de Git: incluye descargas originales,
paneles, matrices, predicciones, figuras y auditorías. Los cuadernos
se distribuyen sin salidas guardadas.

Las direcciones, nombres de archivo y frecuencias se especifican en
`sources.json`. Las fuentes descargables del proyecto son:

| Organismo | Información | Descarga oficial |
| --- | --- | --- |
| MIVAU | valor medio tasado de vivienda por provincia y trimestre. | [mivau_valor_tasado_VDP006_01.csv](https://cdn.mivau.gob.es/portal-web-mivau/Datos_MIVAU/CSV/VDP006_01.csv) |
| MIVAU | Número y valor de transacciones inmobiliarias de vivienda. | [mivau_transacciones_VDP003_01.csv](https://cdn.mivau.gob.es/portal-web-mivau/Datos_MIVAU/CSV/VDP003_01.csv) |
| MIVAU | Viviendas iniciadas y terminadas, libres y protegidas; el fichero abierto usa año y mes. | [mivau_construccion_VDP005_01.csv](https://cdn.mivau.gob.es/portal-web-mivau/Datos_MIVAU/CSV/VDP005_01.csv) |
| INE | Número e importe de hipotecas por provincia y mes. | [76317](https://www.ine.es/wstempus/xlsx/es/DATOS_TABLA/76317) |
| INE | Tasas de actividad, paro y empleo por provincia, sexo y trimestre. | [65349](https://www.ine.es/wstempus/xlsx/es/DATOS_TABLA/65349) |
| INE | Población residente por provincia, fecha, sexo y edad. | [59589](https://www.ine.es/wstempus/xlsx/es/DATOS_TABLA/59589) |
| INE | Población residente española/extranjera por provincia y trimestre. | [59591](https://www.ine.es/wstempus/xlsx/es/DATOS_TABLA/59591) |
| INE | Viajeros y pernoctaciones hoteleras por provincia y mes. | [67190](https://www.ine.es/wstempus/xlsx/es/DATOS_TABLA/67190) |
| INE | Tasas de variación del IPC general por provincia y mes. | [76143](https://www.ine.es/wstempus/xlsx/es/DATOS_TABLA/76143) |
| Banco de España | Euríbor, tipos oficiales del BCE y otros tipos mensuales. | [bde_tipos_interes_mensuales.csv](https://www.bde.es/webbe/es/estadisticas/compartido/datos/csv/cal_umv_mfi001.csv) |
| IGN | Metadatos estáticos: código, superficie, capital, CCAA y costa. | [ign_provincias.xlsx](https://www.ign.es/resources/ane/Informacion_Geografica_Destacada/IGN_INFOGEO_PROVINCIAS.xlsx) |
| INE | Compraventas de vivienda nueva/usada y libre/protegida por provincia y mes. | [6150](https://www.ine.es/wstempus/xlsx/es/DATOS_TABLA/6150) |
| INE | Paro por grupos de edad a nivel CCAA; puede asignarse a sus provincias, etiquetándolo como variable autonómica. | [65334](https://www.ine.es/wstempus/xlsx/es/DATOS_TABLA/65334) |
| MIVAU | Precio medio del suelo por ámbito geográfico, tamaño municipal y trimestre. | [mivau_precio_suelo_VDP004_01.csv](https://cdn.mivau.gob.es/portal-web-mivau/Datos_MIVAU/CSV/VDP004_01.csv) |
| MIVAU | Actuaciones de vivienda protegida y rehabilitación por provincia y mes. | [mivau_vivienda_protegida_VDP007_01.csv](https://cdn.mivau.gob.es/portal-web-mivau/Datos_MIVAU/CSV/VDP007_01.csv) |
| INE | Población provincial anual histórica del Padrón a 1 de enero. | [2852](https://www.ine.es/wstempus/xlsx/es/DATOS_TABLA/2852) |
| INE | Población provincial anual por edad simple, nacionalidad y sexo desde 1998. | [ine_poblacion_edad_historica_03003.csv](https://www.ine.es/jaxi/files/_px/es/csv_bdsc/t20/e245/p08/03003.csv) |
| INE | Censo anual 2021-2025 por provincia, sexo, edad simple y nacionalidad. | [68519](https://www.ine.es/wstempus/xlsx/es/DATOS_TABLA/68519) |
| INE | Renta neta media por persona y hogar por provincia. | [53689](https://www.ine.es/wstempus/xlsx/es/DATOS_TABLA/53689) |
| INE | PIB y VAB por ramas de actividad y provincia. | [76926](https://www.ine.es/wstempus/xlsx/es/DATOS_TABLA/tpx/76926) |
| INE | Empleo total y asalariado por provincia y rama de actividad. | [76927](https://www.ine.es/wstempus/xlsx/es/DATOS_TABLA/tpx/76927) |
| INE | Saldo migratorio exterior e interprovincial por provincia. | [69765](https://www.ine.es/wstempus/xlsx/es/DATOS_TABLA/69765) |

La clasificación `core`/`optional` controla la descarga; para reconstruir
el panel completo se utiliza `--include-optional`. Los nombres esperados
al guardar manualmente cada descarga figuran en `sources.json`.

La [estadística de salarios de la AEAT](https://sede.agenciatributaria.gob.es/Sede/estadisticas/mercado-trabajo-pensiones.html)
figura como referencia manual en el inventario. No forma parte de las
descargas automatizadas ni de los datos utilizados en el panel entregado.

Las fuentes pueden actualizar o revisar sus series. Una descarga nueva
puede producir cifras diferentes a las de la instantánea utilizada en el estudio.

## Instalación y ejecución

El entorno de referencia utiliza Python 3.10. `requirements-lock.txt` fija las
versiones de las dependencias directas; `requirements.txt` indica los mínimos.

Desde la raíz del proyecto, en Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements-lock.txt
.venv\Scripts\python.exe 01_download_sources.py --include-optional
.venv\Scripts\python.exe 02_profile_raw.py
.venv\Scripts\python.exe run_pipeline.py
```

En Linux o macOS, sustituir `.venv\Scripts\python.exe` por `.venv/bin/python`.
El script 01 guarda las descargas en `data/raw/`. Si una dirección ha cambiado,
se puede descargar el archivo desde la fuente oficial y guardarlo con el nombre
indicado en `sources.json`.

Los datos y resultados se generan únicamente en local. El flujo completo vuelve
a ajustar los modelos y puede tardar. Después se pueden ejecutar los cinco
cuadernos para visualizar tablas y gráficos. Sus salidas no se incluyen en la
publicación del código.

`run_pipeline.py` utiliza los ficheros locales y no realiza descargas automáticas.
Para actualizar las fuentes, repetir el script 01 con `--include-optional
--overwrite` y volver a ejecutar las etapas posteriores. El origen final sigue
fijado en 2026Q1; cambiarlo requiere actualizar el protocolo.
`python run_pipeline.py --help` muestra los controles de ejecución.

## Orden de ejecución

| Orden | Script | Resultado |
| --- | --- | --- |
| 1 | `03_build_panel_skeleton.py` | Panel, calendario y disponibilidad temporal |
| 2 | `04_prepare_model_data.py` | Matrices auxiliares y auditoría |
| 3 | `04b_analisis_exploratorio.py` | Exploración y cobertura |
| 4 | `05_benchmark_modelos.py` | Selección conjunta por horizonte |
| 5 | `07_backtest_multihorizonte.py` | Calibración separada de bandas |
| 6 | `08_evaluar_candidatos_finales.py` | Evaluación reciente y referencias |
| 7 | `09_generar_prevision_final.py` | Reajuste y previsión hasta 2028 |
| 8 | `06_analisis_robustez.py` | Diagnósticos descriptivos posteriores |
| 9 | `10_generar_documentacion.py` | Documentación técnica y cuadernos |
| 10 | Pruebas en `tests/` | Comprobaciones de integridad |

`run_pipeline.py` ejecuta esta secuencia. El script 06 se ejecuta después del
09 porque analiza sus salidas. Los scripts 01 y 02 permiten descargar y
perfilar fuentes por separado. Las matrices de `data/model/` son auxiliares;
el modelado principal construye sus objetivos desde el panel.

Para reproducir la extensión híbrida después del flujo principal:

```powershell
.venv\Scripts\python.exe 11_experimento_hibrido.py
```

## Archivos de consulta

- [Protocolo](docs/protocolo_experimental.md): fechas, selección, métricas y limitaciones.
- [Flujo metodológico](docs/flujo_metodologico.md): conexión entre etapas.
- [Diccionario de variables](docs/diccionario_variables.md): fuentes, unidades y disponibilidad.
- [Resultados](docs/resultados.md): selección, evaluación, cobertura y previsión final.
- [Trazabilidad](docs/trazabilidad.md): identificadores e inventario de resultados.
- [Experimento híbrido](docs/experimento_hibrido.md): extensión exploratoria independiente.
- `forecasting/`: implementación de modelos, calendario y etapas.
- `reporting/`: generación de cuadernos, tablas y gráficos.
- `sources.json`: enlaces oficiales y configuración de las descargas.

Las rutas `data/raw/`, `data/processed/`, `data/model/`, `data/reports/` y
`data/forecasts/` se crean en local; su contenido no se publica en GitHub.

Los cinco cuadernos se publican sin salidas y se ejecutan después del flujo:

1. [Preparación y calidad de datos](notebooks/01_preparacion_y_calidad_datos.ipynb).
2. [Análisis de variables](notebooks/02_analisis_de_variables.ipynb).
3. [Selección de modelos y variables](notebooks/03_seleccion_modelos_variables.ipynb).
4. [Evaluación y calibración](notebooks/04_evaluacion_y_calibracion.ipynb).
5. [Previsión hasta 2028](notebooks/05_prevision_hasta_2028.ipynb).

La memoria académica se entrega por separado. El script 10 regenera únicamente
la documentación técnica y los cuadernos, sin ajustar modelos. Las salidas
obtenidas al ejecutar los cuadernos deben mantenerse fuera de la publicación.

## Validación

```powershell
.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Las pruebas completas requieren instalar las dependencias y generar los
datos y resultados locales. Comprueban restricciones del código y coherencia
de artefactos. La capacidad predictiva se valora con las métricas de evaluación.

## Publicación del código

La estructura conserva las rutas necesarias para reproducir el estudio.
`.gitignore` excluye toda la carpeta `data/`, el paquete ZIP de entrega,
entornos, cachés, temporales y configuración personal. Antes de publicar un
cuaderno ejecutado, deben borrarse sus salidas. `.gitattributes` conserva los
bytes del código al clonar en distintos sistemas. Los datos se obtienen
directamente de los organismos oficiales indicados en este README.
