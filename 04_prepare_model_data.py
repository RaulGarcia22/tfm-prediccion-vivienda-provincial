from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
INPUT = ROOT / "data" / "processed" / "panel_provincial_trimestral.parquet"
OUTPUT = ROOT / "data" / "model"
REPORT = ROOT / "data" / "reports" / "model_data_audit.md"

KEYS = ["province_code", "province", "quarter"]
TARGETS = ["target_m2_t1", "target_m2_t4", "target_m2_t8", "target_m2_t12"]
STATIC = ["comunidad_autonoma", "superficie_km2", "longitud_costa_km"]

BASE_FEATURES = [
    "valor_tasado_m2",
    "valor_tasado_m2_lag1",
    "valor_tasado_m2_lag2",
    "valor_tasado_m2_lag4",
    "valor_tasado_m2_qoq",
    "valor_tasado_m2_yoy",
    "tasa_paro",
    "poblacion",
    "densidad_poblacion",
    "poblacion_18_34",
    "pct_poblacion_18_34",
    "crecimiento_poblacion_18_34_yoy",
    "poblacion_25_39",
    "pct_poblacion_25_39",
    "crecimiento_poblacion_25_39_yoy",
    "ipc_yoy",
    "euribor_12m",
    "hipotecas_numero",
    "hipotecas_importe_miles_eur",
    "hipotecas_numero_por_1000_hab",
    "viajeros_hoteles",
    "pernoctaciones_hoteles",
    "viajeros_hoteles_por_1000_hab",
    "pernoctaciones_hoteles_por_1000_hab",
]

ECONOMIC_PREFIXES = (
    "trans_",
    "iniciadas_",
    "terminadas_",
    "compraventas_",
    "vp_",
)
ECONOMIC_EXACT = {
    "precio_suelo_m2",
    "renta_neta_media_persona",
    "renta_neta_media_hogar",
    "pib_miles_eur",
    "vab_construccion_miles_eur",
    "empleo_total_miles_personas",
    "importe_medio_hipoteca_eur",
    "compraventas_por_1000_hab",
    "pib_per_capita_eur",
    "peso_construccion_pib_pct",
    "empleo_sobre_poblacion_pct",
    "precio_m2_sobre_renta_hogar_pct",
    "ratio_precio_suelo_vivienda",
}


def temporal_split(periods: pd.Series) -> pd.Series:
    """Descriptive origin eras only; training must also censor label dates by horizon."""
    split = pd.Series("evaluation", index=periods.index, dtype="string")
    split.loc[periods <= pd.Period("2018Q4", freq="Q")] = "development"
    split.loc[periods.between(pd.Period("2019Q1", freq="Q"), pd.Period("2022Q4", freq="Q"))] = "calibration"
    return split


def existing(columns: list[str], candidates: list[str]) -> list[str]:
    return [column for column in candidates if column in columns]


def save_matrix(panel: pd.DataFrame, name: str, features: list[str]) -> pd.DataFrame:
    columns = KEYS + ["year", "quarter_number", "split"] + features + TARGETS
    matrix = panel[existing(panel.columns.tolist(), columns)].copy()
    matrix.to_parquet(OUTPUT / f"dataset_{name}.parquet", index=False)
    matrix.to_csv(OUTPUT / f"dataset_{name}.csv", index=False)
    return matrix


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    REPORT.parent.mkdir(parents=True, exist_ok=True)

    panel = pd.read_parquet(INPUT)
    periods = pd.PeriodIndex(panel["quarter"], freq="Q")
    panel["year"] = periods.year
    panel["quarter_number"] = periods.quarter
    panel["split"] = temporal_split(pd.Series(periods, index=panel.index))

    forbidden = set(KEYS + TARGETS + ["capital", "province_ign", "split"])
    numeric_features = [
        column
        for column in panel.select_dtypes(include=np.number).columns
        if column not in forbidden
    ]
    base = existing(panel.columns.tolist(), STATIC + BASE_FEATURES)
    economic = [
        column
        for column in numeric_features
        if column.startswith(ECONOMIC_PREFIXES) or column in ECONOMIC_EXACT
    ]
    recent = [column for column in numeric_features if column not in {"year", "quarter_number"}]

    matrices = {
        "base": save_matrix(panel, "base", list(dict.fromkeys(base))),
        "ampliado": save_matrix(panel, "ampliado", list(dict.fromkeys(base + economic))),
        "reciente": save_matrix(panel, "reciente", list(dict.fromkeys(STATIC + recent))),
    }

    lines = [
        "# Auditoría de matrices de modelado",
        "",
        "Etiquetas descriptivas por fecha de origen: desarrollo hasta 2018T4, calibración 2019T1-2022T4 y evaluación desde 2023T1.",
        "Estas etiquetas NO son máscaras suficientes para entrenar: cada horizonte debe excluir etiquetas cuya fecha objetivo aún no se conoce en su origen de evaluación.",
        "El protocolo unificado aplica sus propios cortes por fechas objetivo y origen; estas matrices conservan el calendario íntegro, también cuando falta el precio.",
        "Disponibilidad asumida: exógenas trimestrales t-1; anuales desde año de referencia T4 + 8 trimestres. Véase source_availability_audit.csv.",
        "Los valores ausentes se conservan: cualquier imputación debe ajustarse exclusivamente con entrenamiento.",
        "",
        "## Resumen",
        "",
        "| Matriz | Filas | Variables predictoras | Desarrollo | Calibración | Evaluación |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, matrix in matrices.items():
        counts = matrix["split"].value_counts()
        predictor_count = len(matrix.columns) - len(KEYS) - 3 - len(TARGETS)
        lines.append(
            f"| {name} | {len(matrix):,} | {predictor_count} | "
            f"{counts.get('development', 0):,} | {counts.get('calibration', 0):,} | {counts.get('evaluation', 0):,} |"
        )

    lines += ["", "## Disponibilidad de los objetivos", ""]
    for target in TARGETS:
        lines += [f"### {target}", "", "| Partición | Filas con objetivo |", "|---|---:|"]
        counts = panel.loc[panel[target].notna()].groupby("split").size()
        for split in ("development", "calibration", "evaluation"):
            lines.append(f"| {split} | {counts.get(split, 0):,} |")
        lines.append("")

    lines += [
        "## Cobertura de variables",
        "",
        "| Variable | Cobertura total | Cobertura desarrollo | Primera fecha | Última fecha |",
        "|---|---:|---:|---|---|",
    ]
    for column in recent:
        available = panel[column].notna()
        dated = panel.loc[available, "quarter"]
        total = 100 * available.mean()
        train = 100 * panel.loc[panel["split"].eq("development"), column].notna().mean()
        first = dated.min() if not dated.empty else "-"
        last = dated.max() if not dated.empty else "-"
        lines.append(f"| `{column}` | {total:.1f}% | {train:.1f}% | {first} | {last} |")

    train_mask = panel["split"].eq("development")
    unavailable_train = [column for column in recent if panel.loc[train_mask, column].notna().sum() == 0]
    lines += [
        "",
        "## Alertas para modelado",
        "",
        "- Las matrices son exportaciones descriptivas; la selección unificada trabaja con bloques de variables del panel.",
        "- Un predictor solo se selecciona con datos de desarrollo, cobertura suficiente y validación temporal.",
        "- Las ausencias de precio nunca se rellenan para crear objetivos; los horizontes se alinean con trimestres reales.",
    ]
    if unavailable_train:
        lines.append(
            "- Sin datos en la época de desarrollo: "
            + ", ".join(f"`{column}`" for column in unavailable_train)
            + "."
        )

    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Matrices creadas en: {OUTPUT}")
    for name, matrix in matrices.items():
        print(f"  {name}: {matrix.shape}")
    print(f"Informe: {REPORT}")


if __name__ == "__main__":
    main()
