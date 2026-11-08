"""Regenera la documentación y los cuadernos desde los resultados del protocolo.

No ajusta ni selecciona modelos. Requiere que 03--09 y 06 hayan terminado.
Las cifras se leen de los artefactos; si falta una etapa no se inventan resultados.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from forecasting.direct import AR_FEATURES, EXTERNAL_BLOCKS, FEATURE_BLOCKS, FAMILY_LABELS, ModelSpec
from forecasting.experiment import ROOT, REPORTS, PANEL_PATH, FORECAST_ORIGIN, FORECAST_END
from forecasting.stages import load_selection, validate_evaluation_manifest
from reporting.exploratory_notebooks import quality_notebook, variables_notebook
from reporting.selection_plots import selection_plot_cells
from reporting.hybrid_report import hybrid_cells
from forecasting.unified import BASELINE_LABELS, read_panel


DOCS = ROOT / "docs"
EDA = ROOT / "data/reports/eda"
FORECASTS = ROOT / "data/forecasts"
FORECAST_FILE = FORECASTS / "prevision_provincial_2026T2_2028T4.csv"
LABELS = {**FAMILY_LABELS, **BASELINE_LABELS, "selected": "Selección congelada"}
BLOCK_LABELS = {"ar": "Historia del precio", "credit": "Precio + crédito y operaciones",
                "economy": "Precio + economía", "demography": "Precio + demografía",
                "tourism": "Precio + turismo", "supply": "Precio + oferta",
                "all": "Todos los bloques"}


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def table(frame: pd.DataFrame, columns: list[str] | None = None,
          names: dict[str, str] | None = None, decimals: int = 2) -> str:
    """Tabla Markdown sin depender de tabulate ni de formatos locales."""
    data = frame.loc[:, columns].copy() if columns else frame.copy()
    if names:
        data = data.rename(columns=names)
    if data.empty:
        return "Sin observaciones en este informe."

    def cell(value: object) -> str:
        if pd.isna(value):
            return "—"
        if isinstance(value, (float, np.floating)):
            return f"{value:.{decimals}f}"
        return str(value).replace("|", " / ").replace("\n", " ")

    headers = [str(column) for column in data.columns]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    lines += ["| " + " | ".join(cell(value) for value in row) + " |"
              for row in data.itertuples(index=False, name=None)]
    return "\n".join(lines)


def write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.strip() + "\n", encoding="utf-8")






def read_results() -> dict:
    panel = read_panel(PANEL_PATH)
    selection = load_selection(panel)
    validate_evaluation_manifest(panel, selection)
    # Verificar que están presentes todas las salidas del protocolo vigente.
    paths = {
        "selected": REPORTS / "selected_models.csv",
        "evaluation": REPORTS / "evaluation_metrics.csv",
        "balanced": REPORTS / "evaluation_balanced_metrics.csv",
        "coverage": REPORTS / "interval_coverage.csv",
        "calibration": REPORTS / "interval_calibration.csv",
        "contribution": REPORTS / "feature_block_contribution.csv",
        "uncertainty": REPORTS / "evaluation_paired_uncertainty.csv",
        "coverage_variables": EDA / "cobertura_variables.csv",
        "correlations": EDA / "correlaciones_crecimiento.csv",
        "availability": ROOT / "data/reports/source_availability_audit.csv",
        "forecast": FORECAST_FILE,
    }
    missing = [str(path.relative_to(ROOT)) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Completar las etapas antes de documentar: " + ", ".join(missing))
    data = {key: pd.read_csv(path) for key, path in paths.items()}
    for stage in ("calibration", "evaluation"):
        manifest = read_json(REPORTS / f"{stage}_manifest.json")
        if manifest["selection_id"] != selection["selection_id"]:
            raise ValueError(f"El manifiesto de {stage} no corresponde a esta selección.")
    forecast = data["forecast"]
    if "selection_id" not in forecast or set(forecast.selection_id) != {selection["selection_id"]}:
        raise ValueError("Las previsiones no corresponden a esta selección.")
    # El CSV público utiliza nombres españoles; las tablas y cuadernos usan
    # las mismas etiquetas internas que el evaluador.
    data["forecast"] = forecast.rename(columns={
        "provincia": "province", "trimestre": "forecast_quarter", "prediccion": "prediction",
        "limite_inferior_80": "lower_80", "limite_superior_80": "upper_80",
        "limite_inferior_95": "lower_95", "limite_superior_95": "upper_95",
    })
    data.update(panel=panel, selection=selection, grid=read_json(REPORTS / "candidate_grid.json"),
                data_manifest=read_json(ROOT / "data/reports/data_build_manifest.json"),
                sources=read_json(ROOT / "sources.json"))
    return data


def result_tables(data: dict) -> dict[str, str]:
    winners = data["selected"].copy()
    winners["family"] = winners.family.map(LABELS)
    winners["feature_set"] = winners.feature_set.map(BLOCK_LABELS)
    selected_table = table(winners, ["horizon", "family", "feature_set", "MAE", "origins"],
                           {"horizon": "Horizonte", "family": "Modelo", "feature_set": "Variables",
                            "MAE": "MAE de selección (€/m²)", "origins": "Orígenes"})
    evaluation = data["evaluation"].copy()
    evaluation["role"] = evaluation.role.map(LABELS)
    pivot = evaluation.pivot(index="horizon", columns="role", values="MAE").reset_index()
    pivot.columns.name = None
    pivot = pivot.rename(columns={"horizon": "Horizonte"})
    counts = data["evaluation"].loc[data["evaluation"].role.eq("selected"),
                                    ["horizon", "origins", "observations"]]
    pivot = pivot.merge(counts.rename(columns={"horizon": "Horizonte", "origins": "Orígenes",
                                               "observations": "Casos"}), on="Horizonte")
    balanced = data["balanced"].copy()
    balanced["role"] = balanced.role.map(LABELS)
    balanced_pivot = balanced.pivot(index="horizon", columns="role", values="MAE").reset_index()
    balanced_pivot.columns.name = None
    balanced_pivot = balanced_pivot.rename(columns={"horizon": "Horizonte"})
    coverage = data["coverage"].copy()
    coverage["nominal_coverage"] *= 100
    coverage["empirical_coverage"] *= 100
    coverage_table = table(coverage, ["horizon", "nominal_coverage", "empirical_coverage", "mean_width_eur_m2", "origins"],
                           {"horizon": "Horizonte", "nominal_coverage": "Nominal (%)",
                            "empirical_coverage": "Observada (%)", "mean_width_eur_m2": "Anchura media (€/m²)",
                            "origins": "Orígenes"})
    contributions = data["contribution"].loc[
        data["contribution"].family.isin(["ridge", "huber", "elasticnet"])].copy()
    contributions = contributions.groupby(["family", "feature_set"], as_index=False).agg(
        mejora_media=("improvement_vs_AR_pct", "mean"), horizontes=("horizon", "nunique"))
    contributions["family"] = contributions.family.map(LABELS)
    contributions["feature_set"] = contributions.feature_set.map(BLOCK_LABELS)
    contribution_table = table(contributions, names={"family": "Familia", "feature_set": "Variables",
                               "mejora_media": "Mejora media frente a AR (%)", "horizontes": "Horizontes"})
    forecast = data["forecast"]
    quarter_column = "forecast_quarter" if "forecast_quarter" in forecast else "quarter"
    last = forecast.loc[forecast[quarter_column].eq(FORECAST_END)].copy()
    display = [column for column in ["province", "prediction", "lower_80", "upper_80", "lower_95", "upper_95"] if column in last]
    if "province" in last:
        last = last.sort_values("province")
    forecast_table = table(last, display, {"province": "Territorio", "prediction": "Previsión (€/m²)",
                            "lower_80": "Inferior 80 %", "upper_80": "Superior 80 %",
                            "lower_95": "Inferior 95 %", "upper_95": "Superior 95 %"})
    return {"selection": selected_table, "evaluation": table(pivot), "balanced": table(balanced_pivot),
            "coverage": coverage_table, "contribution": contribution_table, "forecast": forecast_table}


def findings(data: dict) -> dict[str, str]:
    evaluation = data["evaluation"]
    selected = evaluation.loc[evaluation.role.eq("selected")].set_index("horizon")
    annual = evaluation.loc[evaluation.role.eq("annual_trend")].set_index("horizon")
    persistence = evaluation.loc[evaluation.role.eq("persistence")].set_index("horizon")
    count_annual = int((selected.MAE < annual.MAE).sum())
    count_persistence = int((selected.MAE < persistence.MAE).sum())
    ranking = evaluation.groupby("role").MAE.mean().sort_values()
    balanced = data["balanced"].groupby("role").MAE.mean().sort_values()
    coverage95 = data["coverage"].loc[data["coverage"].nominal_coverage.eq(.95)]
    worst = coverage95.loc[coverage95.empirical_coverage.idxmin()]
    shortfall = int((coverage95.empirical_coverage < .95).sum())
    minimum, maximum = 100 * coverage95.empirical_coverage.min(), 100 * coverage95.empirical_coverage.max()
    recent = (f"La selección congelada obtiene menor MAE que persistencia en {count_persistence} de los 12 "
              f"horizontes y que tendencia interanual en {count_annual} de los 12. "
              f"Su MAE medio entre horizontes es {selected.MAE.mean():.2f} €/m². "
              f"El menor promedio retrospectivo entre los roles evaluados corresponde a "
              f"{LABELS.get(ranking.index[0], ranking.index[0])} ({ranking.iloc[0]:.2f} €/m²). "
              "Este promedio da el mismo peso a cada horizonte, aunque los horizontes tienen distintos orígenes; "
              f"sobre los orígenes comunes el menor promedio corresponde a {LABELS.get(balanced.index[0], balanced.index[0])} "
              f"({balanced.iloc[0]:.2f} €/m²).")
    if ranking.index[0] != "selected":
        decision = ("La evaluación posterior no confirma que la selección sea la mejor opción en promedio. "
                    "Se conserva la especificación congelada para que la previsión sea la aplicación del experimento "
                    "realizado; reemplazarla por el ganador de esta evaluación exigiría una nueva selección y una "
                    "evaluación futura. El pronóstico se presenta como resultado experimental, no como superioridad demostrada.")
    else:
        decision = ("La selección tiene el menor promedio en esta evaluación, lo que respalda su uso dentro de este "
                    "diseño. No demuestra superioridad universal ni elimina la incertidumbre de los horizontes largos.")
    interval = (f"Las bandas nominales del 95 % cubren entre {minimum:.1f} % y {maximum:.1f} % de los casos por horizonte; "
                f"{shortfall} de los 12 horizontes quedan por debajo del 95 %. "
                f"La menor cobertura aparece en h={int(worst.horizon)} ({100 * worst.empirical_coverage:.1f} %). ")
    if minimum < 85:
        interval += ("La infracobertura es material: estas bandas no proporcionan protección suficiente frente a la "
                     "dinámica observada en el periodo reciente. Se publican con su cobertura empírica y no como una "
                     "garantía del 95 % para 2028.")
    else:
        interval += ("El nivel nominal es una etiqueta de calibración, no una garantía de cobertura futura; la "
                     "dependencia espacial y temporal y los cambios de distribución limitan su interpretación.")
    winner_counts = data["selected"].groupby(["family", "feature_set"]).size()
    selection = "; ".join(f"{LABELS.get(family, family)} con {BLOCK_LABELS.get(block, block).lower()}: {count} "
                          f"{'horizonte' if count == 1 else 'horizontes'}"
                          for (family, block), count in winner_counts.items()) + "."
    return {"recent": recent, "decision": decision, "interval": interval, "selection": selection}


LIMITATIONS = """
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
"""


def data_sources_markdown(sources: dict) -> str:
    institutions = {"mivau": "MIVAU", "ine": "INE", "bde": "Banco de España", "ign": "IGN"}
    lines = ["## Fuentes oficiales de los datos", "",
             "Los conjuntos de datos no se publican en este repositorio. La carpeta",
             "`data/` completa está excluida de Git: incluye descargas originales,",
             "paneles, matrices, predicciones, figuras y auditorías. Los cuadernos",
             "se distribuyen sin salidas guardadas.", "",
             "Las direcciones, nombres de archivo y frecuencias se especifican en",
             "`sources.json`. Las fuentes descargables del proyecto son:", "",
             "| Organismo | Información | Descarga oficial |",
             "| --- | --- | --- |"]
    for group in ("core", "optional"):
        for name, item in sources[group].items():
            institution = institutions[name.split("_", 1)[0]]
            description = item["description"].removeprefix("Target: ")
            label = str(item.get("table_id", item["filename"]))
            lines.append(f"| {institution} | {description} | [{label}]({item['url']}) |")
    lines.extend(["", "La clasificación `core`/`optional` controla la descarga; para reconstruir",
                  "el panel completo se utiliza `--include-optional`. Los nombres esperados",
                  "al guardar manualmente cada descarga figuran en `sources.json`.", "",
                  "La [estadística de salarios de la AEAT](" + sources["manual"]["aeat_salarios_provinciales"]["url"] + ")",
                  "figura como referencia manual en el inventario. No forma parte de las",
                  "descargas automatizadas ni de los datos utilizados en el panel entregado.", "",
                  "Las fuentes pueden actualizar o revisar sus series. Una descarga nueva",
                  "puede producir cifras diferentes a las de la instantánea utilizada en el estudio."])
    return "\n".join(lines)


def make_readme(data: dict, values: dict, results: dict) -> str:
    manifest = data["data_manifest"]
    return f"""# Predicción provincial del valor tasado de vivienda libre hasta 2028

Código y documentación del estudio de predicción trimestral del valor medio
tasado nominal de vivienda libre en España. La unidad de análisis es
provincia × trimestre: {manifest['territories']} territorios (50 provincias,
Ceuta y Melilla), {manifest['rows']} filas entre {manifest['first_quarter']} y
{manifest['last_quarter']}, con {manifest['observed_target_rows']} valores
observados y {manifest['missing_target_rows']} ausencias explícitas.
La variable se expresa en €/m² nominales.

## Estudio principal y extensión exploratoria

El estudio principal selecciona modelos y variables por horizonte con datos de
desarrollo hasta 2018Q4. La especificación generada en local en
`data/reports/unified/selection.json` se utiliza después en calibración,
evaluación y previsión final. Los modelos se reajustan sin repetir la selección.

El resultado es una **previsión de investigación con origen fijo en 2026Q1**:
{len(data['forecast'])} predicciones para 52 territorios entre 2026Q2 y 2028Q4.
La evaluación es retrospectiva sobre periodos previamente inspeccionados.
No acredita una validación prospectiva ni el funcionamiento de un servicio
operativo de predicción.

La **extensión híbrida del script 11** es una prueba exploratoria posterior que
combina tendencia interanual o Holt con la selección original. Sus resultados
se generan en local en `data/reports/hybrid_experiment/`. No interviene
en la selección, las bandas ni las previsiones finales. Su diseño y resultados
están en [Experimento híbrido](docs/experimento_hibrido.md).

## Resultados principales

{values['selection']}

{values['recent']}

{values['decision']}

{values['interval']}

Las tablas están en [Resultados](docs/resultados.md). Los niveles del 80 % y
95 % de las bandas son nominales; deben interpretarse junto con su cobertura
retrospectiva observada.

{data_sources_markdown(data['sources'])}

## Instalación y ejecución

El entorno de referencia utiliza Python 3.10. `requirements-lock.txt` fija las
versiones de las dependencias directas; `requirements.txt` indica los mínimos.

Desde la raíz del proyecto, en Windows PowerShell:

```powershell
python -m venv .venv
.venv\\Scripts\\python.exe -m pip install -r requirements-lock.txt
.venv\\Scripts\\python.exe 01_download_sources.py --include-optional
.venv\\Scripts\\python.exe 02_profile_raw.py
.venv\\Scripts\\python.exe run_pipeline.py
```

En Linux o macOS, sustituir `.venv\\Scripts\\python.exe` por `.venv/bin/python`.
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
.venv\\Scripts\\python.exe 11_experimento_hibrido.py
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
.venv\\Scripts\\python.exe -m unittest discover -s tests -v
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
"""



def make_protocol(data: dict) -> str:
    selection_id = data["selection"]["selection_id"]
    return f"""# Protocolo experimental vigente

## Objeto y trazabilidad

Se predice el valor medio tasado nominal de vivienda libre provincial, en €/m², a horizontes de 1 a 12 trimestres. La previsión final usa los horizontes 1 a 11, porque 2026Q1 → 2028Q4 contiene once trimestres futuros. El horizonte 12 se conserva para comprobar comportamiento a tres años.

Identificador de selección: `{selection_id}`. Fuente ejecutable: `forecasting/experiment.py`; especificación congelada: `data/reports/unified/selection.json`.

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

{LIMITATIONS}
"""


def make_flow() -> str:
    return """# Conexión de las dos fases

La fase 1 identifica una especificación por horizonte; la fase 2 la aplica tras verificarla y calibrar su incertidumbre. El archivo `selection.json` vincula ambas fases mediante su identificador de contenido.

```mermaid
flowchart TD
    A[Fuentes oficiales y snapshot raw] --> B[03 Calendario y disponibilidad]
    B --> C[04 Matrices auxiliares]
    B --> D[04b Exploración de desarrollo]
    B --> E[05 Selección de modelos y variables]
    D --> E
    E --> F[selection.json congelado por horizonte]
    F --> G[07 Calibración hasta 2022Q4]
    F --> H[08 Evaluación desde 2023Q1]
    G --> H
    F --> I[09 Reajuste en 2026Q1]
    G --> I
    H --> J[Informes de error y cobertura]
    I --> K[Previsión provincial hasta 2028Q4]
    J --> L[06 Robustez descriptiva]
    K --> L
    L --> M[10 Documentación y notebooks]
```

## Fase 1: aprender qué funciona en desarrollo

El panel conserva todos los trimestres y no interpola precios ausentes. Las fuentes externas se alinean con sus retrasos supuestos antes de calcular ratios. El análisis exploratorio informa cobertura, asociaciones y redundancia. Las correlaciones no sustituyen la prueba predictiva.

La búsqueda compara modelos y conjuntos de variables usando cuatro fechas objetivo históricas. Un cribado a 1, 4, 8 y 12 trimestres limita el coste. Cuatro finalistas de familias distintas compiten con todas las referencias en los doce horizontes. Se congela la mejor especificación de cada horizonte.

## Fase 2: evaluar y aplicar lo seleccionado

La calibración usa exclusivamente el periodo anterior a 2023. La evaluación mide los modelos congelados con datos posteriores y referencias bajo el mismo contrato de origen. Un resultado desfavorable se mantiene visible y no causa una selección encubierta.

La previsión final ajusta esos mismos modelos con etiquetas conocidas en 2026Q1. Los modelos directos reciben las variables observadas del origen y producen cada horizonte sin introducir variables externas futuras. Si un horizonte seleccionó la referencia recursiva, sus valores intermedios son predicciones, nunca precios reales futuros.

El resultado contiene provincia, origen, trimestre previsto, horizonte, modelo, bloque, identificador de selección, predicción y bandas. Los manifiestos y auditorías relacionan las cifras con la selección y con el conjunto de datos.

## Lectura de artefactos

| Necesidad | Artefacto |
| --- | --- |
| Saber qué modelo y variables se eligieron | `data/reports/unified/selection.json` y `selected_models.csv` |
| Revisar la parrilla completa | `candidate_grid.json`, `screening_metrics.csv` |
| Ver el aporte de bloques | `feature_block_contribution.csv` |
| Examinar errores recientes | `evaluation_metrics.csv`, `evaluation_province.csv`, `evaluation_origin.csv` |
| Comparar horizontes en los mismos orígenes | `evaluation_balanced_metrics.csv` |
| Revisar incertidumbre y cobertura | `interval_calibration.csv`, `interval_coverage.csv` |
| Ver sensibilidad | `robustness_*.csv` y `forecast_trajectory_diagnostics.csv` |
| Consultar el pronóstico | `data/forecasts/prevision_provincial_2026T2_2028T4.csv` |

Los archivos relativos sin prefijo de la tabla están en `data/reports/unified`. Los cuadernos activos solo leen el flujo vigente.
"""


def make_dictionary(data: dict) -> str:
    panel, availability = data["panel"], data["availability"]
    membership = {name: ", ".join(block for block, features in FEATURE_BLOCKS.items() if name in features)
                  for name in FEATURE_BLOCKS["all"]}
    units = {"price_log": "log(€/m²)", "quarter_sin": "adimensional", "quarter_cos": "adimensional",
             "growth_1q": "100 × log(P[t]/P[t−1])", "growth_4q": "100 × log(P[t]/P[t−4])",
             "growth_lag1": "100 × log(P[t−1]/P[t−2])", "growth_lag4": "100 × log(P[t−4]/P[t−5])",
             "volatility_4q": "Desviación típica de cuatro crecimientos log trimestrales",
             "province_code": "Código INE de dos caracteres", "quarter": "Periodo trimestral",
             "province": "Nombre territorial", "valor_tasado_m2": "€/m² nominales",
             "importe_medio_hipoteca_eur": "Importe total (miles €) × 1000 / número de hipotecas",
             "pib_per_capita_eur": "PIB (miles €) × 1000 / población disponible",
             "peso_construccion_pib_pct": "100 × VAB construcción / PIB",
             "empleo_sobre_poblacion_pct": "100000 × empleo (miles) / población disponible",
             "densidad_poblacion": "Habitantes / km²", "euribor_12m": "% anual, promedio trimestral",
             "tasa_paro": "%", "ipc_yoy": "% interanual", "precio_suelo_m2": "€/m² de suelo",
             "superficie_km2": "km²", "longitud_costa_km": "km"}
    rows = []
    audit = availability.drop_duplicates("variable").set_index("variable")
    for column in panel:
        if column.startswith("target_"):
            usage, note = "Etiqueta auxiliar, nunca predictor", "Precio en t+h por fecha exacta; sin imputación"
        elif column in membership:
            usage, note = membership[column], "Predictor de origen; la inclusión efectiva depende del entrenamiento"
        elif column in ("province_code", "quarter"):
            usage, note = "Clave; provincia se codifica en modelos directos", "No es una magnitud continua"
        elif column == "valor_tasado_m2":
            usage, note = "Objetivo y base de transformaciones AR", "Publicado en el origen; sin imputación del objetivo"
        else:
            usage, note = "Auxiliar/descriptiva; fuera de la parrilla final", "No confundir presencia en el panel con uso por el modelo"
        if column in audit.index:
            rule = str(audit.loc[column, "availability_rule"])
            component = str(audit.loc[column, "component"])
        else:
            rule, component = note, "Derivada o identificador"
        unit = units.get(column, "")
        if not unit:
            if "por_1000_hab" in column:
                unit = "Numerador disponible × 1000 / población disponible"
            elif column.startswith(("pct_", "crecimiento_")) or column.endswith(("_yoy", "_qoq")):
                unit = "%"
            elif column.endswith("miles_eur") or column.startswith("trans_valor_"):
                unit = "Miles de euros; consultar concepto de fuente"
            elif column.startswith("poblacion"):
                unit = "Personas"
            elif column.startswith("target_") or column.startswith("valor_tasado_m2_lag"):
                unit = "€/m²"
            elif column.startswith("renta_neta"):
                unit = "Euros anuales"
            elif column == "empleo_total_miles_personas":
                unit = "Miles de personas"
            elif column.startswith(("saldo_migratorio", "hipotecas_numero", "compraventas_", "iniciadas_", "terminadas_", "vp_", "trans_numero_", "viajeros", "pernoctaciones")):
                unit = "Recuento del concepto de la fuente"
            else:
                unit = "Texto/categoría; consultar fuente"
        rows.append({"Variable": column, "Unidad o fórmula": unit, "Origen": component,
                     "Cobertura (%)": 100 * panel[column].notna().mean(), "Uso": usage,
                     "Disponibilidad": rule})
    ar = pd.DataFrame([{"Variable": name, "Definición": units[name]} for name in AR_FEATURES])
    blocks = pd.DataFrame([{"Conjunto": block, "Descripción": BLOCK_LABELS[block],
                           "Variables numéricas": len(features), "Externas añadidas": ", ".join(EXTERNAL_BLOCKS.get(block, [])) or ("Unión de los cinco bloques" if block == "all" else "Ninguna")}
                          for block, features in FEATURE_BLOCKS.items()])
    source_rows = []
    for group, sources in data["sources"].items():
        for name, source in sources.items():
            path = ROOT / "data/raw" / source.get("filename", "__not_a_file__")
            url = source.get("url", "")
            source_rows.append({"Fuente": f"[{name}]({url})" if url else name,
                                "Descripción": source.get("description", ""), "Grupo": group,
                                "Archivo presente": "Sí" if path.is_file() else "No"})
    return f"""# Diccionario de datos y variables del experimento

## Fuentes

El inventario ejecutable es `sources.json`. Esta tabla distingue fuentes registradas de archivos presentes durante la ejecución local; los ficheros de datos no se distribuyen en GitHub. La tabla de disponibilidad identifica las variables incorporadas al panel. Una fuente manual u opcional registrada no acredita que se haya utilizado como predictor.

{table(pd.DataFrame(source_rows))}

## Variables del modelo directo

Todas las especificaciones directas incluyen el bloque AR y la provincia como categoría. Las columnas completamente vacías se excluyen dentro de cada entrenamiento. Las variables externas ya están alineadas por disponibilidad cuando el modelo las recibe.

{table(ar)}

`volatility_4q` usa desviación poblacional (`ddof=0`). Seno y coseno son funciones de `2π × número_trimestre / 4`. Los crecimientos logarítmicos del modelo no son idénticos a las variaciones porcentuales simples auxiliares del panel.

{table(blocks)}

La variable dependiente es `100 × log(P[t+h]/P[t])`, para h=1,…,12. Se devuelve a euros mediante `P[t] × exp(predicción/100)`. Los retardos y objetivos usan trimestres exactos: si falta t−1, no se sustituye por la última fila existente.

## Disponibilidad

Los datos anuales entran en el cuarto trimestre de su año de referencia más ocho trimestres y se arrastran únicamente hacia delante. Las fuentes mensuales/trimestrales externas entran un trimestre después de su periodo de referencia. Las series recientes de población trimestral sustituyen la anual retrasada cuando están disponibles. Los ratios se calculan después de ese alineamiento, de modo que pueden combinar magnitudes con distintas antigüedades informativas.

`source_availability_audit.csv` conserva el supuesto por componente. Sus extremos de disponibilidad describen el componente antes del recorte al panel, y pueden superar 2026Q1; no significan que el pronóstico haya leído ese futuro. La cobertura siguiente se calcula sobre las filas efectivas del panel.

## Panel completo

{table(pd.DataFrame(rows))}

## Cobertura de predictores de la parrilla

{table(data['coverage_variables'].loc[data['coverage_variables'].variable.isin(FEATURE_BLOCKS['all'])], ['variable', 'primera_observacion', 'cobertura_desarrollo_pct', 'cobertura_train_inicial_pct', 'cobertura_total_pct'], {'variable': 'Variable', 'primera_observacion': 'Primera disponible', 'cobertura_desarrollo_pct': 'Desarrollo (%)', 'cobertura_train_inicial_pct': 'Hasta 2010Q4 (%)', 'cobertura_total_pct': 'Panel (%)'})}

Las tasas de cobertura distintas implican que las correlaciones pueden utilizar muestras diferentes. La imputación permite ajustar modelos, pero no crea evidencia observada donde una fuente todavía no existía.
"""






def make_results(data: dict, values: dict, results: dict) -> str:
    return f"""# Resultados del estudio principal

Selección: `{data['selection']['selection_id']}`. Las tablas proceden de
`data/reports/unified/` y de la previsión final en `data/forecasts/`.
Este informe técnico acompaña al código; la memoria académica se entrega aparte.

## Interpretación de la evaluación

{values['recent']}

{values['decision']}

{values['interval']}

## Especificaciones seleccionadas

Los errores de esta tabla corresponden a desarrollo y no a la evaluación reciente.

{results['selection']}

## MAE de evaluación por horizonte

Los métodos se comparan sobre los mismos casos dentro de cada horizonte.
El número de orígenes cambia entre horizontes.

{results['evaluation']}

## MAE sobre orígenes comunes a todos los horizontes

{results['balanced']}

## Cobertura retrospectiva de las bandas

{results['coverage']}

## Aportación de bloques en desarrollo

La mejora compara el mejor ajuste de cada bloque con el mejor ajuste AR de
su familia y horizonte. No identifica efectos causales de las variables.

{results['contribution']}

## Previsiones provinciales

El script 09 genera en local las previsiones desde 2026Q1 hasta 2028Q4,
manteniendo la especificación elegida para cada horizonte. Las tablas
individuales y las figuras se guardan en `data/forecasts/` y no se publican.

## Extensión híbrida

La combinación de referencias con los modelos seleccionados se documenta
por separado en [Experimento híbrido](experimento_hibrido.md).
Se trata de una prueba exploratoria posterior y no genera esta previsión.
"""


def make_traceability(data: dict) -> str:
    reports = sorted(path.name for path in REPORTS.iterdir() if path.is_file())
    specs = [row for row in data["grid"] if row["family"] not in BASELINE_LABELS]
    config_rows = [{"Familia": LABELS.get(family, family),
                    "Configuraciones": sum(row["family"] == family for row in specs)}
                   for family in sorted({row["family"] for row in specs})]
    return f"""# Trazabilidad de los resultados

Los identificadores siguientes vinculan los resultados con los datos, la
implementación y las especificaciones del estudio principal.

## Identificadores

- Selección: `{data['selection']['selection_id']}`.
- Datos de desarrollo: `{data['selection']['development_data_hash']}`.
- Implementación de selección: `{data['selection']['implementation_hash']}`.
- Origen final de previsión: `{FORECAST_ORIGIN}`.
- Último trimestre previsto: `{FORECAST_END}`.

## Configuraciones comparadas

{table(pd.DataFrame(config_rows))}

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

{chr(10).join('- `' + name + '`' for name in reports)}

Estos archivos se generan en local en `data/reports/unified/` y no se publican
en el repositorio. Los datos brutos y sus
huellas están descritos en `data/reports/data_build_manifest.json`; las reglas
de disponibilidad figuran en `data/reports/source_availability_audit.csv`.
La configuración final en `data/forecasts/` contiene las huellas de sus entradas
y salidas. Git conserva los bytes originales mediante `.gitattributes` para
mantener estas comprobaciones al clonar en distintos sistemas.
"""



def notebook_cell(kind: str, text: str) -> dict:
    cell = {"cell_type": kind, "metadata": {}, "source": text.strip().splitlines(keepends=True)}
    if kind == "code":
        cell.update(execution_count=None, outputs=[])
    return cell


def make_notebooks(only: set[str] | None = None) -> None:
    directory = ROOT / "notebooks"
    directory.mkdir(exist_ok=True)
    setup = """from pathlib import Path
import json
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display, Image
ROOT = Path.cwd()
if not (ROOT / 'sources.json').is_file():
    ROOT = ROOT.parent
assert (ROOT / 'sources.json').is_file(), 'Abrir el cuaderno desde el repositorio o notebooks/'
REPORTS = ROOT / 'data/reports/unified'
EDA = ROOT / 'data/reports/eda'
pd.set_option('display.max_columns', 20)
"""
    notebooks = {
        "01_preparacion_y_calidad_datos.ipynb": quality_notebook(setup),
        "02_analisis_de_variables.ipynb": variables_notebook(setup),
        "03_seleccion_modelos_variables.ipynb": [
            ("markdown", "# Selección conjunta de modelos y variables\nLeer la parrilla, el cribado y la selección congelada por horizonte. No se utilizan los resultados de evaluación reciente para cambiar los ganadores."),
            ("code", setup),
            ("code", "selection = json.loads((REPORTS / 'selection.json').read_text(encoding='utf-8'))\ndisplay(selection['protocol'])\ngrid = pd.DataFrame(json.loads((REPORTS / 'candidate_grid.json').read_text(encoding='utf-8')))\ndisplay(grid.groupby(['family', 'feature_set']).size().rename('configuraciones').to_frame())"),
            ("code", "display(pd.read_csv(REPORTS / 'screening_ranking.csv').head(20))\ndisplay(pd.read_csv(REPORTS / 'selected_models.csv'))\ndisplay(selection['selected_by_horizon'])"),
            ("markdown", "## Aportación predictiva de bloques\nMejora frente a AR calculada dentro de familia y horizonte, optimizando la configuración de cada bloque. Son cifras de selección; no efectos causales ni pruebas de estabilidad futura."),
            ("code", "blocks = pd.read_csv(REPORTS / 'feature_block_contribution.csv')\ndisplay(blocks[blocks.family.isin(['ridge', 'huber', 'elasticnet'])].pivot_table(index=['family', 'feature_set'], columns='horizon', values='improvement_vs_AR_pct'))\nmetrics = pd.read_csv(REPORTS / 'selection_metrics.csv')\nmetrics.pivot(index='horizon', columns='candidate_id', values='MAE').plot(figsize=(12, 5), ylabel='MAE de selección (€/m²)')\nplt.show()"),
        ],
        "04_evaluacion_y_calibracion.ipynb": [
            ("markdown", "# Evaluación posterior y cobertura\nLa selección está fijada. Calibración hasta 2022Q4 y objetivos de evaluación desde 2023Q1. La evaluación es retrospectiva sobre una historia que ya se había inspeccionado."),
            ("code", setup),
            ("code", "metrics = pd.read_csv(REPORTS / 'evaluation_metrics.csv')\ndisplay(metrics)\nmetrics.pivot(index='horizon', columns='role', values='MAE').plot(marker='o', figsize=(11, 5), ylabel='MAE (€/m²)')\nplt.show()\ndisplay(pd.read_csv(REPORTS / 'evaluation_balanced_metrics.csv'))"),
            ("markdown", "## Bandas marginales\nContrastar cobertura nominal con observada. Los horizontes largos tienen pocos orígenes; el número de provincias no equivale a observaciones independientes. Infracobertura implica que las bandas deben interpretarse con especial cautela."),
            ("code", "coverage = pd.read_csv(REPORTS / 'interval_coverage.csv')\ndisplay(coverage)\nax = coverage.pivot(index='horizon', columns='nominal_coverage', values='empirical_coverage').plot(marker='o', ylim=(0, 1), figsize=(10, 4))\nax.axhline(.8, linestyle='--', color='grey')\nax.axhline(.95, linestyle='--', color='grey')\nax.set_ylabel('Cobertura empírica')\nplt.show()\ndisplay(pd.read_csv(REPORTS / 'evaluation_paired_uncertainty.csv'))"),
            ("code", "for name in ['robustness_temporal.csv', 'robustness_territorial.csv', 'robustness_selection_stability.csv']:\n    path = REPORTS / name\n    if path.is_file():\n        print(name)\n        display(pd.read_csv(path).head(30))"),
        ],
        "05_prevision_hasta_2028.ipynb": [
            ("markdown", "# Aplicación de los modelos seleccionados hasta 2028\nOrigen fijo 2026Q1. Las bandas tienen niveles nominales del 80 % y 95 %; consultar su cobertura reciente antes de interpretarlas. Cada horizonte hereda la especificación de la fase 1."),
            ("code", setup),
            ("code", "forecast = pd.read_csv(ROOT / 'data/forecasts/prevision_provincial_2026T2_2028T4.csv', dtype={'province_code': str}).rename(columns={'provincia': 'province', 'trimestre': 'forecast_quarter', 'prediccion': 'prediction', 'limite_inferior_80': 'lower_80', 'limite_superior_80': 'upper_80', 'limite_inferior_95': 'lower_95', 'limite_superior_95': 'upper_95'})\nselection = json.loads((REPORTS / 'selection.json').read_text(encoding='utf-8'))\nif 'selection_id' in forecast:\n    assert set(forecast.selection_id) == {selection['selection_id']}\ndisplay(forecast.head())\ndisplay(forecast[forecast.forecast_quarter.eq('2028Q4')])"),
            ("code", "province_code = '28'  # Cambiar por otro código INE de dos caracteres\npart = forecast.loc[forecast.province_code.str.zfill(2).eq(province_code)].sort_values('horizon')\nax = part.plot(x='horizon', y='prediction', marker='o', figsize=(10, 5), legend=False)\nfor confidence, alpha in [(95, .15), (80, .25)]:\n    low, high = f'lower_{confidence}', f'upper_{confidence}'\n    if low in part and high in part:\n        ax.fill_between(part.horizon, part[low], part[high], alpha=alpha, label=f'Banda nominal {confidence} %')\nax.set(title=str(part.province.iloc[0]), xlabel='Trimestres desde 2026Q1', ylabel='Valor tasado previsto (€/m²)')\nax.legend()\nplt.show()"),
            ("markdown", "## Coherencia de la trayectoria\nLos modelos independientes por horizonte pueden generar cambios entre puntos consecutivos. No se suavizan después de evaluar. Esta previsión no incorpora información publicada después del origen fijo ni describe un intervalo nacional."),
            ("code", "path = REPORTS / 'forecast_trajectory_diagnostics.csv'\nif path.is_file():\n    display(pd.read_csv(path))"),
        ],
    }
    notebooks['03_seleccion_modelos_variables.ipynb'].extend(selection_plot_cells())
    notebooks['03_seleccion_modelos_variables.ipynb'].extend(hybrid_cells('development'))
    notebooks['04_evaluacion_y_calibracion.ipynb'].extend(hybrid_cells('evaluation'))
    if only is not None and not only.issubset(notebooks):
        raise ValueError('Se ha solicitado un cuaderno ajeno al flujo vigente.')
    for name, cells in notebooks.items():
        if only is not None and name not in only:
            continue
        notebook = {"cells": [notebook_cell(kind, text) for kind, text in cells],
                    "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                 "language_info": {"name": "python", "version": "3"}},
                    "nbformat": 4, "nbformat_minor": 4}
        write(directory / name, json.dumps(notebook, ensure_ascii=False, indent=2))


def main() -> None:
    data = read_results()
    results, values = result_tables(data), findings(data)
    outputs = {
        ROOT / "README.md": make_readme(data, values, results),
        DOCS / "protocolo_experimental.md": make_protocol(data),
        DOCS / "flujo_metodologico.md": make_flow(),
        DOCS / "diccionario_variables.md": make_dictionary(data),
        DOCS / "resultados.md": make_results(data, values, results),
        DOCS / "trazabilidad.md": make_traceability(data),
    }
    for path, content in outputs.items():
        write(path, content)
        print(f"[DOC] {path.relative_to(ROOT)}", flush=True)
    make_notebooks()
    print("[DOC] Cinco cuadernos de lectura del flujo vigente; selección " + data["selection"]["selection_id"], flush=True)


if __name__ == "__main__":
    main()
