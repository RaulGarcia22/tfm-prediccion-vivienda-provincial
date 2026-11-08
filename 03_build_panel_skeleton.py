from __future__ import annotations

import csv
import hashlib
import json
import re
import unicodedata
from functools import lru_cache, reduce
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from forecasting.calendar import calendar_shift, complete_quarterly_grid

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
COMPONENTS = PROCESSED / "components"
REPORTS = ROOT / "data" / "reports"
ANNUAL_PUBLICATION_DELAY = 8
QUARTERLY_PUBLICATION_DELAY = 1


def snake(value: object) -> str:
    text = str(value).strip()
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = re.sub(r"[^0-9a-zA-Z]+", "_", text).strip("_").lower()
    return text


def read_csv_flexible(path: Path) -> pd.DataFrame:
    raw = path.read_bytes()[:100_000]
    encoding = "utf-8-sig"
    try:
        raw.decode(encoding)
    except UnicodeDecodeError:
        encoding = "latin-1"
    sample = raw.decode(encoding, errors="replace")
    try:
        delimiter = csv.Sniffer().sniff(sample, delimiters=";,\t|").delimiter
    except csv.Error:
        delimiter = ";"
    return pd.read_csv(path, sep=delimiter, encoding=encoding, low_memory=False)


def read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".xlsx":
        df = pd.read_excel(path)
    elif path.suffix.lower() == ".csv":
        df = read_csv_flexible(path)
    else:
        raise ValueError(f"Formato no soportado: {path}")
    df.columns = [snake(c) for c in df.columns]
    return df


def numeric(series: pd.Series, dot_thousands: bool = False) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce")
    text = series.astype("string").str.strip()
    text = text.replace({"": np.nan, "..": np.nan, ".": np.nan, "nan": np.nan})
    if dot_thousands:
        text = text.str.replace(".", "", regex=False).str.replace(",", ".", regex=False)
    else:
        # El INE usa puntos decimales en XLSX y algunos CSV usan formato español.
        has_comma = text.str.contains(",", regex=False, na=False)
        text.loc[has_comma] = (
            text.loc[has_comma]
            .str.replace(".", "", regex=False)
            .str.replace(",", ".", regex=False)
        )
    return pd.to_numeric(text, errors="coerce")


def province_code(series: pd.Series) -> pd.Series:
    extracted = series.astype(str).str.extract(r"^\s*(\d{1,2})\b", expand=False)
    return extracted.str.zfill(2)


MONTHS = {
    "enero": 1, "febrero": 2, "marzo": 3, "abril": 4,
    "mayo": 5, "junio": 6, "julio": 7, "agosto": 8,
    "septiembre": 9, "setiembre": 9, "octubre": 10,
    "noviembre": 11, "diciembre": 12,
    "ene": 1, "feb": 2, "mar": 3, "abr": 4, "jun": 6,
    "jul": 7, "ago": 8, "sep": 9, "oct": 10, "nov": 11, "dic": 12,
}


def parse_period(value: object) -> pd.Period | pd.NaT:
    if pd.isna(value):
        return pd.NaT
    raw = str(value).strip().lower()
    plain = snake(raw).replace("_", " ")

    # 2024T1, 2024 T1, T1 2024, 1T 2024
    m = re.search(r"\b(19|20)\d{2}\s*[-_/ ]?\s*t\s*([1-4])\b", plain)
    if m:
        year = int(re.search(r"(19|20)\d{2}", plain).group())
        quarter = int(re.search(r"t\s*([1-4])", plain).group(1))
        return pd.Period(f"{year}Q{quarter}", freq="Q")
    m = re.search(r"\bt\s*([1-4])\s*[-_/ ]?\s*((?:19|20)\d{2})\b", plain)
    if m:
        return pd.Period(f"{int(m.group(2))}Q{int(m.group(1))}", freq="Q")
    m = re.search(r"\b([1-4])\s*t\s*((?:19|20)\d{2})\b", plain)
    if m:
        return pd.Period(f"{int(m.group(2))}Q{int(m.group(1))}", freq="Q")

    # 2024M01, 2024-01, 01/2024.
    for pattern in (
        r"\b((?:19|20)\d{2})\s*m\s*(0?[1-9]|1[0-2])\b",
        r"\b((?:19|20)\d{2})[-_/](0?[1-9]|1[0-2])\b",
        r"\b(0?[1-9]|1[0-2])[-_/]((?:19|20)\d{2})\b",
    ):
        m = re.search(pattern, plain)
        if m:
            a, b = int(m.group(1)), int(m.group(2))
            year, month = (a, b) if a > 100 else (b, a)
            return pd.Period(f"{year}-{month:02d}", freq="M").asfreq("Q")

    # Enero 2024 / 2024 Enero.
    year_match = re.search(r"\b((?:19|20)\d{2})\b", plain)
    if year_match:
        year = int(year_match.group(1))
        for name, month in MONTHS.items():
            if name in plain:
                return pd.Period(f"{year}-{month:02d}", freq="M").asfreq("Q")

    # Fechas trimestrales del padrón: 1 de enero/abril/julio/octubre de 2024.
    m = re.search(
        r"\b1\s+de\s+(enero|abril|julio|octubre)\s+de\s+((?:19|20)\d{2})\b",
        plain,
    )
    if m:
        return pd.Period(
            f"{int(m.group(2))}-{MONTHS[m.group(1)]:02d}", freq="M"
        ).asfreq("Q")

    return pd.NaT


def add_quarter(df: pd.DataFrame, period_col: str) -> pd.DataFrame:
    out = df.copy()
    out["quarter"] = out[period_col].map(parse_period)
    return out.dropna(subset=["quarter"])


def find_col(df: pd.DataFrame, *tokens: str) -> str:
    tokens = tuple(snake(t) for t in tokens)
    matches = [c for c in df.columns if all(token in c for token in tokens)]
    if not matches:
        raise KeyError(f"No se encontró columna con tokens {tokens}. Columnas: {list(df.columns)}")
    return matches[0]


def filter_contains(df: pd.DataFrame, column: str, text: str) -> pd.DataFrame:
    return df[df[column].astype(str).map(snake).str.contains(snake(text), na=False)]


def save_component(df: pd.DataFrame, name: str) -> Path:
    COMPONENTS.mkdir(parents=True, exist_ok=True)
    path = COMPONENTS / f"{name}.parquet"
    sort_keys = [c for c in ("province_code", "quarter") if c in df.columns]
    df.sort_values(sort_keys).to_parquet(path, index=False)
    print(f"[OK] {name}: {df.shape} -> {path}")
    return path


SINGLE_PROVINCE_AUTONOMIES = {
    3: "33", 4: "07", 6: "39", 13: "28", 14: "30", 15: "31", 17: "26",
}


def mivau_province_codes(df: pd.DataFrame, code_col: str) -> pd.Series:
    """Normaliza CPRO y recupera provincias uniprovinciales publicadas como total CCAA."""
    codes = numeric(df[code_col]).astype("Int64")
    if "codauto" in df.columns:
        auto = numeric(df["codauto"]).astype("Int64")
        for auto_code, province in SINGLE_PROVINCE_AUTONOMIES.items():
            condition = (codes.isna() & auto.eq(auto_code)).fillna(False)
            codes = codes.mask(condition, int(province))
    return codes.astype("Int64").astype(str).str.zfill(2).replace("<NA>", np.nan)


@lru_cache(maxsize=1)
def province_reference() -> dict[str, str]:
    """Códigos y nombres oficiales del IGN, usados para excluir filas de CCAA/total."""
    df = pd.read_excel(RAW / "ign_provincias.xlsx", sheet_name="IGN_INFOGEO_PROVINCIAS")
    df.columns = [snake(c) for c in df.columns]
    return {
        str(int(code)).zfill(2): str(name)
        for code, name in zip(df["identificador"], df["nombre"])
        if pd.notna(code) and pd.notna(name)
    }


def province_rows(labels: pd.Series) -> pd.Series:
    """Detecta únicamente provincias; evita confundir códigos de CCAA con CPRO."""
    reference = province_reference()

    def is_province(value: object) -> bool:
        match = re.match(r"^\s*(\d{2})\s+(.+?)\s*$", str(value))
        if not match or match.group(1) not in reference:
            return False
        expected = set(snake(reference[match.group(1)]).split("_"))
        observed = set(snake(match.group(2)).split("_"))
        return expected == observed or expected.issubset(observed) or observed.issubset(expected)

    return labels.map(is_province)


def province_code_from_name(value: object) -> str | None:
    observed = set(snake(value).split("_"))
    for code, name in province_reference().items():
        expected = set(snake(name).split("_"))
        if expected == observed:
            return code
    return None


def read_ine_block(
    path: Path, *header_tokens: str, row_section: str | None = None
) -> pd.DataFrame:
    """Extrae un bloque de las hojas matriciales del INE y lo devuelve en largo."""
    raw = pd.read_excel(path, header=None)
    period_scores = raw.apply(
        lambda row: sum(not pd.isna(parse_period(value)) for value in row), axis=1
    )
    period_row = int(period_scores.idxmax())
    if period_scores.loc[period_row] < 3:
        raise ValueError(f"No se detectó la fila de periodos en {path.name}")

    header = raw.iloc[:period_row].ffill(axis=1)
    wanted = pd.Series(True, index=raw.columns)
    for token in header_tokens:
        token_norm = snake(token)
        wanted &= header.apply(
            lambda col: col.astype(str).map(snake).str.contains(token_norm, regex=False).any()
        )
    period_cols = [
        col for col in raw.columns
        if wanted.loc[col] and not pd.isna(parse_period(raw.loc[period_row, col]))
    ]
    if not period_cols:
        raise ValueError(
            f"No se encontró el bloque {header_tokens} en {path.name}"
        )

    body = raw.iloc[period_row + 1:].copy()
    if row_section:
        labels = raw.iloc[:, 0].astype(str).map(snake)
        starts = raw.index[labels.eq(snake(row_section))]
        if starts.empty:
            raise ValueError(f"No se encontró la sección {row_section!r} en {path.name}")
        start = int(starts[0]) + 1
        end = len(raw)
        for idx in range(start, len(raw)):
            label = raw.iloc[idx, 0]
            if (
                pd.notna(label)
                and str(label).strip()
                and raw.iloc[idx, 1:].isna().all()
                and not province_rows(pd.Series([label])).iloc[0]
            ):
                end = idx
                break
        body = raw.iloc[start:end].copy()
    body = body.loc[province_rows(body.iloc[:, 0]), [0] + period_cols]
    body = body.rename(columns={0: "province"})
    long = body.melt(id_vars="province", var_name="_column", value_name="value")
    periods = {col: parse_period(raw.loc[period_row, col]) for col in period_cols}
    long["quarter"] = long["_column"].map(periods)
    long["province_code"] = province_code(long["province"])
    long["value"] = numeric(long["value"])
    return long.dropna(subset=["province_code", "quarter", "value"])


def process_target() -> pd.DataFrame:
    """MIVAU VDP006_01. Columnas documentadas oficialmente."""
    path = RAW / "mivau_valor_tasado_VDP006_01.csv"
    df = read_table(path)

    regime_col = find_col(df, "regimen")
    df = filter_contains(df, regime_col, "libre")

    year_col = find_col(df, "ano")
    quarter_col = find_col(df, "trimestre")
    value_col = find_col(df, "valor")
    code_col = find_col(df, "cpro")
    province_col = find_col(df, "provincia")

    out = pd.DataFrame({
        "province_code": mivau_province_codes(df, code_col),
        "province": df[province_col].astype(str).str.strip(),
        "quarter": [
            pd.Period(f"{int(y)}Q{int(q)}", freq="Q")
            for y, q in zip(numeric(df[year_col]), numeric(df[quarter_col]))
        ],
        "valor_tasado_m2": numeric(df[value_col]),
    })
    out["province"] = out["province_code"].map(province_reference()).fillna(out["province"])
    out.loc[out["valor_tasado_m2"].le(0), "valor_tasado_m2"] = np.nan
    return out.dropna(subset=["province_code", "valor_tasado_m2"]).drop_duplicates(
        ["province_code", "quarter"], keep="last"
    )


@lru_cache(maxsize=1)
def panel_end_quarter() -> pd.Period:
    """End determined by the actual target source, never a hardcoded date."""
    return process_target()["quarter"].max()



def process_transactions() -> pd.DataFrame:
    """
    MIVAU VDP003_01.

    El fichero se transforma de forma genérica: todas las dimensiones que no
    sean tiempo, territorio o valor se combinan para crear nombres de variables.
    Así se conservan, por ejemplo, número/valor, vivienda nueva/usada y
    libre/protegida, según las categorías presentes en la versión descargada.
    """
    path = RAW / "mivau_transacciones_VDP003_01.csv"
    df = read_table(path)

    year_col = find_col(df, "ano")
    quarter_col = find_col(df, "trimestre")
    code_col = find_col(df, "codprovincia")
    type_col = find_col(df, "tipo")
    number_col = find_col(df, "numero", "transacciones")
    amount_col = find_col(df, "valor", "transacciones")

    df["province_code"] = numeric(df[code_col]).astype("Int64").astype(str).str.zfill(2)
    df["quarter"] = [
        pd.Period(f"{int(y)}Q{int(q)}", freq="Q")
        for y, q in zip(numeric(df[year_col]), numeric(df[quarter_col]))
    ]

    values = df.melt(
        id_vars=["province_code", "quarter", type_col],
        value_vars=[number_col, amount_col],
        var_name="measure",
        value_name="value_num",
    )
    values["feature"] = (
        "trans_" + values["measure"].map(snake) + "_" + values[type_col].map(snake)
    )
    values["value_num"] = numeric(values["value_num"])
    pivot = (
        values.pivot_table(
            index=["province_code", "quarter"],
            columns="feature",
            values="value_num",
            aggfunc="sum",
        )
        .reset_index()
    )
    pivot.columns.name = None
    return pivot

def process_construction() -> pd.DataFrame:
    """MIVAU VDP005_01. Columnas documentadas oficialmente."""
    path = RAW / "mivau_construccion_VDP005_01.csv"
    df = read_table(path)

    year_col = find_col(df, "ano")
    month_col = find_col(df, "mes", "numero")
    type_col = find_col(df, "tipo", "vivienda")
    state_col = find_col(df, "estado")
    code_col = find_col(df, "cpro")
    value_col = find_col(df, "valor")

    df["province_code"] = mivau_province_codes(df, code_col)
    df["quarter"] = [
        pd.Period(f"{int(y)}-{int(m):02d}", freq="M").asfreq("Q")
        for y, m in zip(numeric(df[year_col]), numeric(df[month_col]))
    ]
    df["feature"] = (
        df[state_col].map(snake) + "_" + df[type_col].map(snake)
    )
    df["value_num"] = numeric(df[value_col])

    pivot = (
        df.pivot_table(
            index=["province_code", "quarter"],
            columns="feature",
            values="value_num",
            aggfunc="sum",
        )
        .reset_index()
    )
    pivot.columns.name = None
    return pivot.dropna(subset=["province_code"])


def ine_to_long(path: Path) -> pd.DataFrame:
    """
    Convierte una exportación XLSX del INE a formato largo.

    El INE puede entregar:
      1) una columna 'periodo' y una o varias columnas de medida; o
      2) periodos como columnas.
    """
    df = read_table(path)

    period_candidates = [c for c in df.columns if "periodo" in c or c in {"fecha", "mes"}]
    if period_candidates:
        period_col = period_candidates[0]
        dimension_cols = []
        measure_cols = []
        for col in df.columns:
            if col == period_col:
                continue
            converted = numeric(df[col])
            coverage = converted.notna().mean()
            if coverage >= 0.50:
                measure_cols.append(col)
            else:
                dimension_cols.append(col)
        if not measure_cols:
            raise ValueError(f"No se detectaron columnas de medida en {path.name}")
        long = df.melt(
            id_vars=dimension_cols + [period_col],
            value_vars=measure_cols,
            var_name="_measure",
            value_name="value",
        ).rename(columns={period_col: "_period"})
    else:
        period_cols = [c for c in df.columns if not pd.isna(parse_period(c))]
        if not period_cols:
            raise ValueError(
                f"No se detectó columna de periodo ni columnas temporales en {path.name}"
            )
        dimension_cols = [c for c in df.columns if c not in period_cols]
        long = df.melt(
            id_vars=dimension_cols,
            value_vars=period_cols,
            var_name="_period",
            value_name="value",
        )
        long["_measure"] = "value"

    long["value"] = numeric(long["value"])
    long["quarter"] = long["_period"].map(parse_period)
    return long.dropna(subset=["quarter", "value"])


def detect_province_column(df: pd.DataFrame) -> str:
    candidates = [c for c in df.columns if "provincia" in c]
    if not candidates:
        raise KeyError(f"No se encontró una dimensión provincial: {list(df.columns)}")
    return candidates[0]


def process_epa() -> pd.DataFrame:
    raw = pd.read_excel(RAW / "ine_epa_provincial_65349.xlsx", header=None)
    period_row, measure_row = 7, 6
    headers = raw.loc[measure_row].dropna()
    start = next(int(col) for col, value in headers.items() if "tasa_de_paro" in snake(value))
    following = [int(col) for col in headers.index if int(col) > start]
    end = min(following) if following else len(raw.columns)
    period_cols = [col for col in range(start, end) if pd.notna(parse_period(raw.loc[period_row, col]))]
    labels = raw.iloc[:, 0].astype(str).map(snake)
    section_start = int(raw.index[labels.eq("ambos_sexos")][0]) + 1
    section_end = int(raw.index[labels.eq("hombres")][0])
    body = raw.iloc[section_start:section_end]
    body = body.loc[province_rows(body.iloc[:, 0]), [0] + period_cols].rename(columns={0: "province"})
    out = body.melt(id_vars="province", var_name="_column", value_name="tasa_paro")
    out["province_code"] = province_code(out["province"])
    out["quarter"] = out["_column"].map({col: parse_period(raw.loc[period_row, col]) for col in period_cols})
    out["tasa_paro"] = numeric(out["tasa_paro"])
    return out[["province_code", "quarter", "tasa_paro"]].dropna().drop_duplicates(
        ["province_code", "quarter"]
    )


@lru_cache(maxsize=1)
def annual_census_by_age() -> pd.DataFrame:
    """Censo anual 2021-2025: total provincial por edad simple y ambos sexos."""
    raw = pd.read_excel(RAW / "ine_censo_anual_edad_68519.xlsx", header=None)
    header = raw.iloc[6:10].ffill(axis=1)
    selected_columns = []
    ages = {}
    years = {}
    for col in raw.columns[1:]:
        if snake(header.loc[6, col]) != "total" or snake(header.loc[8, col]) != "total":
            continue
        age_label = str(header.loc[7, col])
        if snake(age_label) == "todas_las_edades":
            age = -1
        else:
            match = re.match(r"^\s*(\d{1,3})\s+", age_label)
            if not match:
                continue
            age = int(match.group(1))
        year = pd.to_numeric(pd.Series([raw.loc[9, col]]), errors="coerce").iloc[0]
        if pd.isna(year):
            continue
        selected_columns.append(col)
        ages[col] = age
        years[col] = int(year)

    records = []
    geography = raw.iloc[10:].copy()
    geography = geography.loc[province_rows(geography.iloc[:, 0])]
    for idx in geography.index:
        code = province_code(pd.Series([raw.iloc[idx, 0]])).iloc[0]
        for col in selected_columns:
            records.append(
                {
                    "province_code": code,
                    "year": years[col],
                    "age": ages[col],
                    "value": numeric(pd.Series([raw.loc[idx, col]])).iloc[0],
                }
            )
    return pd.DataFrame(records).dropna(subset=["province_code", "year", "value"])


def process_population() -> pd.DataFrame:
    path = RAW / "ine_poblacion_59589.xlsx"
    raw = pd.read_excel(path, header=None)
    period_row = 6
    period_cols = [c for c in raw.columns if not pd.isna(parse_period(raw.loc[period_row, c]))]
    all_ages = raw.index[raw.iloc[:, 0].astype(str).map(snake).eq("todas_las_edades")]
    if all_ages.empty:
        raise ValueError("No se encontró el bloque 'Todas las edades' de población")
    records = []
    for idx in range(int(all_ages[0]) + 1, len(raw) - 1):
        label = raw.iloc[idx, 0]
        if not province_rows(pd.Series([label])).iloc[0]:
            continue
        value_row = idx + 1
        if snake(raw.iloc[value_row, 0]) != "total":
            continue
        for col in period_cols:
            records.append((label, raw.loc[period_row, col], raw.loc[value_row, col]))
    df = pd.DataFrame(records, columns=["province", "_period", "value"])
    df["province_code"] = province_code(df["province"])
    df["quarter"] = df["_period"].map(parse_period)
    df["poblacion"] = numeric(df["value"])
    recent = df[["province_code", "quarter", "poblacion"]].dropna().drop_duplicates(
        ["province_code", "quarter"]
    )

    history_records = []
    historical_path = RAW / "ine_poblacion_edad_historica_03003.csv"
    for chunk in pd.read_csv(historical_path, sep=";", dtype=str, chunksize=200_000):
        province_col, age_col, nationality_col, sex_col, year_col, value_col = chunk.columns[:6]
        selected = chunk[
            chunk[age_col].str.casefold().eq("total edades")
            & chunk[nationality_col].eq("TOTAL")
            & chunk[sex_col].str.casefold().eq("ambos sexos")
        ].copy()
        selected["province_code"] = province_code(selected[province_col])
        selected["year"] = pd.to_numeric(selected[year_col], errors="coerce")
        selected["poblacion"] = numeric(selected[value_col], dot_thousands=True)
        history_records.append(selected[["province_code", "year", "poblacion"]])
    historical = pd.concat(history_records, ignore_index=True).dropna()
    census = annual_census_by_age()
    census = census[census["age"].eq(-1)].copy()
    census = census.rename(columns={"value": "poblacion"})[
        ["province_code", "year", "poblacion"]
    ]
    historical = pd.concat([historical, census], ignore_index=True).drop_duplicates(
        ["province_code", "year"], keep="last"
    )
    return combine_population_releases(historical, recent, ["poblacion"])


def process_young_population() -> pd.DataFrame:
    """Población de 18-34 y 25-39 años: Padrón anual + ECP trimestral."""
    historical_parts = []
    path = RAW / "ine_poblacion_edad_historica_03003.csv"
    for chunk in pd.read_csv(path, sep=";", dtype=str, chunksize=200_000):
        province_col, age_col, nationality_col, sex_col, year_col, value_col = chunk.columns[:6]
        ages = pd.to_numeric(chunk[age_col].str.extract(r"(\d+)", expand=False), errors="coerce")
        selected = chunk[
            chunk[nationality_col].eq("TOTAL")
            & chunk[sex_col].str.casefold().eq("ambos sexos")
            & ages.between(18, 39)
        ].copy()
        if selected.empty:
            continue
        selected["age"] = ages.loc[selected.index]
        selected["province_code"] = province_code(selected[province_col])
        selected["year"] = pd.to_numeric(selected[year_col], errors="coerce")
        selected["value"] = numeric(selected[value_col], dot_thousands=True)
        selected["poblacion_18_34"] = selected["value"].where(selected["age"].between(18, 34), 0)
        selected["poblacion_25_39"] = selected["value"].where(selected["age"].between(25, 39), 0)
        historical_parts.append(
            selected.groupby(["province_code", "year"], as_index=False)[
                ["poblacion_18_34", "poblacion_25_39"]
            ].sum(min_count=1)
        )

    historical = pd.concat(historical_parts, ignore_index=True).groupby(
        ["province_code", "year"], as_index=False
    )[["poblacion_18_34", "poblacion_25_39"]].sum(min_count=1)
    census = annual_census_by_age()
    census = census[census["age"].between(18, 39)].copy()
    census["poblacion_18_34"] = census["value"].where(census["age"].between(18, 34), 0)
    census["poblacion_25_39"] = census["value"].where(census["age"].between(25, 39), 0)
    census = census.groupby(["province_code", "year"], as_index=False)[
        ["poblacion_18_34", "poblacion_25_39"]
    ].sum(min_count=1)
    historical = pd.concat([historical, census], ignore_index=True).drop_duplicates(
        ["province_code", "year"], keep="last"
    )
    raw = pd.read_excel(RAW / "ine_poblacion_59589.xlsx", header=None)
    period_cols = [col for col in raw.columns[1:] if not pd.isna(parse_period(raw.loc[6, col]))]
    age_headers = []
    for idx in raw.index[:-1]:
        match = re.match(r"^\s*(\d{1,3})\s+", str(raw.iloc[idx, 0]))
        if match and snake(raw.iloc[idx + 1, 0]) == "total_nacional":
            age_headers.append((int(idx), int(match.group(1))))

    records = []
    for position, (start, age) in enumerate(age_headers):
        if not 18 <= age <= 39:
            continue
        end = age_headers[position + 1][0] if position + 1 < len(age_headers) else len(raw)
        for idx in range(start + 1, end - 1):
            label = raw.iloc[idx, 0]
            if not province_rows(pd.Series([label])).iloc[0]:
                continue
            if snake(raw.iloc[idx + 1, 0]) != "total":
                continue
            code = province_code(pd.Series([label])).iloc[0]
            for col in period_cols:
                records.append({
                    "province_code": code,
                    "quarter": parse_period(raw.loc[6, col]),
                    "age": age,
                    "value": numeric(pd.Series([raw.loc[idx + 1, col]])).iloc[0],
                })

    recent_rows = pd.DataFrame(records)
    recent_rows["poblacion_18_34"] = recent_rows["value"].where(
        recent_rows["age"].between(18, 34), 0
    )
    recent_rows["poblacion_25_39"] = recent_rows["value"].where(
        recent_rows["age"].between(25, 39), 0
    )
    recent = recent_rows.groupby(["province_code", "quarter"], as_index=False)[
        ["poblacion_18_34", "poblacion_25_39"]
    ].sum(min_count=1)

    values = ["poblacion_18_34", "poblacion_25_39"]
    return combine_population_releases(historical, recent, values)


def process_foreign_population() -> pd.DataFrame:
    path = RAW / "ine_poblacion_extranjera_59591.xlsx"
    raw = pd.read_excel(path, header=None)
    period_row = 6
    period_cols = [c for c in raw.columns if not pd.isna(parse_period(raw.loc[period_row, c]))]
    geo_indices = [
        int(idx) for idx in raw.index
        if province_rows(pd.Series([raw.iloc[idx, 0]])).iloc[0]
    ]
    records = []
    for position, idx in enumerate(geo_indices):
        end = geo_indices[position + 1] if position + 1 < len(geo_indices) else len(raw)
        block = raw.iloc[idx + 1:end]
        all_ages = block.index[block.iloc[:, 0].astype(str).map(snake).eq("todas_las_edades")]
        if all_ages.empty:
            continue
        after_age = block.loc[int(all_ages[0]) + 1:]
        foreign = after_age.index[after_age.iloc[:, 0].astype(str).map(snake).eq("extranjera")]
        if foreign.empty:
            continue
        value_row = int(foreign[0]) + 1
        for col in period_cols:
            records.append((raw.iloc[idx, 0], raw.loc[period_row, col], raw.loc[value_row, col]))
    df = pd.DataFrame(records, columns=["province", "_period", "value"])
    df["province_code"] = province_code(df["province"])
    df["quarter"] = df["_period"].map(parse_period)
    df["poblacion_extranjera"] = numeric(df["value"])
    return df[["province_code", "quarter", "poblacion_extranjera"]].dropna().drop_duplicates(
        ["province_code", "quarter"]
    )


def process_cpi() -> pd.DataFrame:
    df = read_ine_block(RAW / "ine_ipc_provincial_76143.xlsx", "variacion anual")

    # La tabla incluye tipos de tasa. Priorizamos variación anual.
    for col in [c for c in df.columns if "tipo" in c or "tasa" in c]:
        values = df[col].astype(str).map(snake)
        if values.str.contains("anual").any():
            df = df[values.str.contains("anual")]
            break

    out = (
        df.groupby(["province_code", "quarter"], as_index=False)["value"]
        .mean()
        .rename(columns={"value": "ipc_yoy"})
    )
    return out.dropna(subset=["province_code"])


def process_tourism() -> pd.DataFrame:
    raw = pd.read_excel(RAW / "ine_turismo_hoteles_67190.xlsx", header=None)
    measure_row, category_row, period_row = 6, 7, 8
    measure_headers = raw.loc[measure_row].dropna()
    category_headers = raw.loc[category_row].dropna()
    frames = []
    for token, name in [("viajero", "viajeros_hoteles"), ("pernoctaciones", "pernoctaciones_hoteles")]:
        start = next(int(col) for col, value in measure_headers.items() if token in snake(value))
        following_categories = [int(col) for col in category_headers.index if int(col) > start]
        end = min(following_categories) if following_categories else len(raw.columns)
        period_cols = [col for col in range(start, end) if pd.notna(parse_period(raw.loc[period_row, col]))]
        body = raw.iloc[period_row + 1:].copy()
        body = body.loc[province_rows(body.iloc[:, 0]), [0] + period_cols].rename(columns={0: "province"})
        long = body.melt(id_vars="province", var_name="_column", value_name="value")
        long["province_code"] = province_code(long["province"])
        long["quarter"] = long["_column"].map(
            {col: parse_period(raw.loc[period_row, col]) for col in period_cols}
        )
        long["value"] = numeric(long["value"])
        frames.append(
            long.groupby(["province_code", "quarter"], as_index=False)["value"]
            .sum(min_count=1)
            .rename(columns={"value": name})
        )
    return frames[0].merge(frames[1], on=["province_code", "quarter"], validate="one_to_one")

    # Compatibilidad histórica con exportaciones tabulares del INE.
    df = ine_to_long(RAW / "ine_turismo_hoteles_67190.xlsx")
    province_col = detect_province_column(df)
    df["province_code"] = province_code(df[province_col])

    # El concepto puede aparecer como dimensión o como nombre de medida.
    concept_col = None
    for col in df.columns:
        values = df[col].astype(str).map(snake)
        if values.str.contains("viajeros").any() and values.str.contains("pernoctaciones").any():
            concept_col = col
            break

    if concept_col:
        df["concept"] = df[concept_col].map(snake)
    else:
        df["concept"] = df["_measure"].map(snake)

    grouped = (
        df.groupby(["province_code", "quarter", "concept"], as_index=False)["value"]
        .sum()
    )
    pivot = grouped.pivot_table(
        index=["province_code", "quarter"],
        columns="concept",
        values="value",
        aggfunc="sum",
    ).reset_index()
    pivot.columns.name = None

    rename = {}
    for col in pivot.columns:
        if "viajer" in col:
            rename[col] = "viajeros_hoteles"
        elif "pernoct" in col:
            rename[col] = "pernoctaciones_hoteles"
    return pivot.rename(columns=rename)


def process_mortgages() -> pd.DataFrame:
    path = RAW / "ine_hipotecas_76317.xlsx"
    frames = []
    for measure, name in [
        ("numero de hipotecas", "hipotecas_numero"),
        ("importe de hipotecas", "hipotecas_importe_miles_eur"),
    ]:
        block = read_ine_block(path, "viviendas", measure)
        frames.append(
            block.groupby(["province_code", "quarter"], as_index=False)["value"]
            .sum()
            .rename(columns={"value": name})
        )
    return frames[0].merge(
        frames[1], on=["province_code", "quarter"], validate="one_to_one"
    )

    # Compatibilidad histórica con exportaciones tabulares del INE.
    df = ine_to_long(RAW / "ine_hipotecas_76317.xlsx")
    province_col = detect_province_column(df)
    df["province_code"] = province_code(df[province_col])

    # Solo viviendas.
    nature_cols = [
        c for c in df.columns
        if "naturaleza" in c or "finca" in c or "inmueble" in c
    ]
    for col in nature_cols:
        values = df[col].astype(str).map(snake)
        if values.str.contains("vivienda").any():
            df = df[values.str.contains("vivienda")]
            break

    # El concepto puede ser una dimensión (Hipotecas/Miles €) o columnas de medida.
    concept_col = None
    for col in df.columns:
        values = df[col].astype(str).map(snake)
        if values.str.contains("hipoteca").any() and values.str.contains("miles").any():
            concept_col = col
            break

    if concept_col:
        df["concept"] = df[concept_col].map(snake)
    else:
        df["concept"] = df["_measure"].map(snake)

    grouped = (
        df.groupby(["province_code", "quarter", "concept"], as_index=False)["value"]
        .sum()
    )
    pivot = grouped.pivot_table(
        index=["province_code", "quarter"],
        columns="concept",
        values="value",
        aggfunc="sum",
    ).reset_index()
    pivot.columns.name = None

    rename = {}
    for col in pivot.columns:
        if col in {"province_code", "quarter"}:
            continue
        if "miles" in col or "importe" in col or "capital" in col:
            rename[col] = "hipotecas_importe_miles_eur"
        elif "hipoteca" in col or "numero" in col:
            rename[col] = "hipotecas_numero"
    return pivot.rename(columns=rename)



def process_bde_euribor() -> pd.DataFrame:
    """
    Extrae Euríbor a 12 meses del CSV mensual del Banco de España.

    El catálogo puede entregar el fichero en dos orientaciones:
      - fechas en filas y series en columnas;
      - series en filas y fechas en columnas.
    """
    path = RAW / "bde_tipos_interes_mensuales.csv"
    raw = pd.read_csv(path, sep=",", encoding="latin-1", header=None, low_memory=False)
    description_rows = raw.index[
        raw.iloc[:, 0].astype(str).map(snake).str.contains("descripcion_de_la_serie")
    ]
    if description_rows.empty:
        raise ValueError("No se encontró la fila de descripciones del catálogo BdE")
    descriptions = raw.loc[int(description_rows[0])].astype(str).map(snake)
    candidates = descriptions.index[
        descriptions.str.contains("euribor") & descriptions.str.contains("12_meses")
    ]
    if candidates.empty:
        raise ValueError("No se localizó la serie Euríbor a 12 meses en el CSV del BdE")
    series_col = int(candidates[0])
    temp = pd.DataFrame({"_period": raw.iloc[:, 0], "value": numeric(raw.iloc[:, series_col])})
    temp["quarter"] = temp["_period"].map(parse_period)
    return (
        temp.dropna(subset=["quarter", "value"])
        .groupby("quarter", as_index=False)["value"]
        .mean()
        .rename(columns={"value": "euribor_12m"})
    )

    # Compatibilidad histórica con otras orientaciones del catálogo.
    df = read_table(path)

    # Caso A: una columna de fecha y una columna cuyo nombre contiene Euribor 1 año.
    date_cols = [
        c for c in df.columns
        if any(token in c for token in ("fecha", "periodo", "date", "mes"))
    ]
    euribor_cols = [
        c for c in df.columns
        if "euribor" in c and any(token in c for token in ("1_ano", "12_meses", "un_ano"))
    ]
    if date_cols and euribor_cols:
        temp = pd.DataFrame({
            "_period": df[date_cols[0]],
            "value": numeric(df[euribor_cols[0]]),
        })
    else:
        # Caso B: localizar la fila de la serie y fundir columnas temporales.
        text_df = df.astype(str).apply(lambda col: col.map(snake))
        row_mask = pd.Series(False, index=df.index)
        for col in text_df.columns:
            row_mask |= (
                text_df[col].str.contains("euribor", na=False)
                & text_df[col].str.contains(r"1_ano|12_meses|un_ano", regex=True, na=False)
            )
        candidates = df[row_mask]
        if candidates.empty:
            # Búsqueda menos restrictiva: cualquier fila Euribor y posterior
            # inspección manual mediante raw_profile.md.
            for col in text_df.columns:
                row_mask |= text_df[col].str.contains("euribor", na=False)
            candidates = df[row_mask]
        if candidates.empty:
            raise ValueError("No se localizó ninguna serie Euríbor en el CSV del BdE.")

        row = candidates.iloc[0]
        period_cols = [c for c in df.columns if not pd.isna(parse_period(c))]
        if not period_cols:
            raise ValueError(
                "Se encontró la serie Euríbor, pero no se detectaron columnas temporales."
            )
        temp = pd.DataFrame({
            "_period": period_cols,
            "value": [numeric(pd.Series([row[c]])).iloc[0] for c in period_cols],
        })

    temp["quarter"] = temp["_period"].map(parse_period)
    out = (
        temp.dropna(subset=["quarter", "value"])
        .groupby("quarter", as_index=False)["value"]
        .mean()
        .rename(columns={"value": "euribor_12m"})
    )
    return out


def process_ign_metadata() -> pd.DataFrame:
    """Metadatos estáticos de provincias: superficie, costa, CCAA y capital."""
    path = RAW / "ign_provincias.xlsx"
    df = pd.read_excel(path, sheet_name="IGN_INFOGEO_PROVINCIAS")
    df.columns = [snake(c) for c in df.columns]
    code_col = find_col(df, "identificador")
    name_col = find_col(df, "nombre")

    out = pd.DataFrame({
        "province_code": numeric(df[code_col]).astype("Int64").astype(str).str.zfill(2),
        "province_ign": df[name_col].astype(str).str.strip(),
    })

    for source_tokens, target in [
        (("superficie",), "superficie_km2"),
        (("capital",), "capital"),
        (("comunidad", "autonoma"), "comunidad_autonoma"),
        (("longitud", "costa"), "longitud_costa_km"),
    ]:
        try:
            col = find_col(df, *source_tokens)
            if target in {"superficie_km2", "longitud_costa_km"}:
                out[target] = numeric(df[col])
            else:
                out[target] = df[col].astype(str).str.strip()
        except KeyError:
            pass

    return out.dropna(subset=["province_code"]).drop_duplicates("province_code")


def expand_annual_component(
    df: pd.DataFrame, value_columns: list[str], publication_delay_quarters: int
) -> pd.DataFrame:
    """Expande observaciones anuales desde su trimestre razonable de disponibilidad."""
    work = df.dropna(subset=["province_code", "year"]).copy()
    work = work.drop_duplicates(["province_code", "year"], keep="last")
    for column in value_columns:
        work[column] = pd.to_numeric(work[column], errors="coerce")
    work["quarter"] = [
        pd.Period(f"{int(year)}Q4", freq="Q") + publication_delay_quarters
        for year in work["year"]
    ]
    work = work.sort_values(["province_code", "quarter"])
    end = panel_end_quarter()
    frames = []
    for code, group in work.groupby("province_code"):
        start = group["quarter"].min()
        index = pd.period_range(start, end, freq="Q")
        expanded = group.set_index("quarter")[value_columns].reindex(index).ffill()
        expanded["province_code"] = code
        expanded["quarter"] = index
        frames.append(expanded.reset_index(drop=True))
    return pd.concat(frames, ignore_index=True)[["province_code", "quarter"] + value_columns]


def delay_quarterly_component(df: pd.DataFrame, quarters: int = QUARTERLY_PUBLICATION_DELAY) -> pd.DataFrame:
    """Move a reference quarter to its assumed availability quarter, without filling."""
    out = df.copy()
    out["quarter"] = pd.PeriodIndex(out["quarter"], freq="Q") + quarters
    return out


def combine_population_releases(
    annual: pd.DataFrame, quarterly: pd.DataFrame, values: list[str]
) -> pd.DataFrame:
    """Combine released annual/quarterly stocks without treating years as Q1 data.

    Annual values are available at reference-year Q4 + 8 quarters. Quarterly
    values are available at reference quarter + 1. Once a newer quarterly stock
    is available it takes precedence over the older annual stock. Stocks are
    carried forward only after that release; no backwards filling is performed.
    """
    annual_expanded = expand_annual_component(annual, values, ANNUAL_PUBLICATION_DELAY)
    recent = delay_quarterly_component(quarterly)
    frames = []
    for code, group in recent.groupby("province_code"):
        index = pd.period_range(group["quarter"].min(), panel_end_quarter(), freq="Q")
        expanded = group.set_index("quarter")[values].sort_index().reindex(index).ffill()
        expanded["province_code"] = code
        expanded["quarter"] = index
        frames.append(expanded.reset_index(drop=True))
    if not frames:
        return annual_expanded
    keys = ["province_code", "quarter"]
    recent_expanded = pd.concat(frames, ignore_index=True).set_index(keys)
    combined = recent_expanded[values].combine_first(annual_expanded.set_index(keys)[values])
    return combined.reset_index().sort_values(keys)[keys + values]


def process_land_price() -> pd.DataFrame:
    df = read_table(RAW / "mivau_precio_suelo_VDP004_01.csv")
    df = filter_contains(df, find_col(df, "tipo"), "PMS General")
    code_col = find_col(df, "cpro")
    df["province_code"] = mivau_province_codes(df, code_col)
    df["quarter"] = [
        pd.Period(f"{int(y)}Q{int(q)}", freq="Q")
        for y, q in zip(numeric(df[find_col(df, "ano")]), numeric(df[find_col(df, "trimestre")]))
    ]
    df["precio_suelo_m2"] = numeric(df[find_col(df, "valor")])
    return df[["province_code", "quarter", "precio_suelo_m2"]].dropna().drop_duplicates(
        ["province_code", "quarter"]
    )


def process_protected_housing() -> pd.DataFrame:
    df = read_table(RAW / "mivau_vivienda_protegida_VDP007_01.csv")
    code_col = find_col(df, "cpro")
    df["province_code"] = mivau_province_codes(df, code_col)
    df["quarter"] = [
        pd.Period(f"{int(y)}-{int(m):02d}", freq="M").asfreq("Q")
        for y, m in zip(numeric(df[find_col(df, "ano")]), numeric(df[find_col(df, "mes")]))
    ]
    df["feature"] = (
        "vp_" + df[find_col(df, "tipologia")].map(snake)
        + "_" + df[find_col(df, "estado")].map(snake)
    )
    df["value"] = numeric(df[find_col(df, "numero", "viviendas")])
    out = df.pivot_table(
        index=["province_code", "quarter"], columns="feature", values="value", aggfunc="sum"
    ).reset_index()
    out.columns.name = None
    return out.dropna(subset=["province_code"])


def process_deed_sales() -> pd.DataFrame:
    path = RAW / "ine_compraventas_6150.xlsx"
    raw = pd.read_excel(path, header=None)
    period_row = 6
    period_cols = [c for c in raw.columns if not pd.isna(parse_period(raw.loc[period_row, c]))]
    province_indices = [
        int(i) for i in raw.index if province_rows(pd.Series([raw.iloc[i, 0]])).iloc[0]
    ]
    records = []
    for position, idx in enumerate(province_indices):
        end = province_indices[position + 1] if position + 1 < len(province_indices) else len(raw)
        block = raw.iloc[idx + 1:end]
        for row_idx in block.index:
            concept = snake(raw.iloc[row_idx, 0])
            if concept not in {
                "viviendas_total", "vivienda_nueva", "vivienda_usada",
                "vivienda_libre", "vivienda_protegida",
            }:
                continue
            for col in period_cols:
                records.append((raw.iloc[idx, 0], raw.loc[period_row, col], concept, raw.loc[row_idx, col]))
    df = pd.DataFrame(records, columns=["province", "_period", "concept", "value"])
    df["province_code"] = province_code(df["province"])
    df["quarter"] = df["_period"].map(parse_period)
    df["value"] = numeric(df["value"])
    out = df.pivot_table(
        index=["province_code", "quarter"], columns="concept", values="value", aggfunc="sum"
    ).reset_index()
    out.columns = [c if c in {"province_code", "quarter"} else f"compraventas_{c}" for c in out.columns]
    return out


def process_income() -> pd.DataFrame:
    raw = pd.read_excel(RAW / "ine_renta_hogares_53689.xlsx", header=None)
    province_indices = [
        int(i) for i in raw.index if province_code_from_name(raw.iloc[i, 0]) is not None
    ]
    records = []
    for position, idx in enumerate(province_indices):
        end = province_indices[position + 1] if position + 1 < len(province_indices) else len(raw)
        block = raw.iloc[idx + 1:end]
        measures = {
            "renta_neta_media_por_persona": "renta_neta_media_persona",
            "renta_neta_media_por_hogar": "renta_neta_media_hogar",
        }
        current = None
        yearly: dict[int, dict[str, float]] = {}
        for row_idx in block.index:
            label = snake(raw.iloc[row_idx, 0])
            if label in measures:
                current = measures[label]
                continue
            if current and re.fullmatch(r"(?:19|20)\d{2}", str(raw.iloc[row_idx, 0]).strip()):
                year = int(raw.iloc[row_idx, 0])
                yearly.setdefault(year, {})[current] = numeric(pd.Series([raw.iloc[row_idx, 1]])).iloc[0]
        for year, values in yearly.items():
            records.append({"province_code": province_code_from_name(raw.iloc[idx, 0]), "year": year, **values})
    df = pd.DataFrame(records)
    return expand_annual_component(
        df, ["renta_neta_media_persona", "renta_neta_media_hogar"], 8
    )


def process_gdp() -> pd.DataFrame:
    raw = pd.read_excel(RAW / "ine_pib_provincial_76926.xlsx", header=None)
    years = {c: int(re.search(r"(?:19|20)\d{2}", str(raw.loc[6, c])).group())
             for c in raw.columns if re.search(r"(?:19|20)\d{2}", str(raw.loc[6, c]))}
    province_indices = [int(i) for i in raw.index if province_rows(pd.Series([raw.iloc[i, 0]])).iloc[0]]
    records = []
    for position, idx in enumerate(province_indices):
        end = province_indices[position + 1] if position + 1 < len(province_indices) else len(raw)
        block = raw.iloc[idx + 1:end]
        pib_rows = block.index[block.iloc[:, 0].astype(str).map(snake).str.contains("producto_interior_bruto_a_precios_de_mercado")]
        construction_rows = block.index[block.iloc[:, 0].astype(str).map(snake).eq("f_construccion")]
        if pib_rows.empty:
            continue
        for col, year in years.items():
            record = {
                "province_code": province_code(pd.Series([raw.iloc[idx, 0]])).iloc[0],
                "year": year,
                "pib_miles_eur": numeric(pd.Series([raw.loc[int(pib_rows[0]), col]])).iloc[0],
            }
            if not construction_rows.empty:
                record["vab_construccion_miles_eur"] = numeric(
                    pd.Series([raw.loc[int(construction_rows[0]), col]])
                ).iloc[0]
            records.append(record)
    return expand_annual_component(
        pd.DataFrame(records), ["pib_miles_eur", "vab_construccion_miles_eur"], 8
    )


def process_regional_employment() -> pd.DataFrame:
    raw = pd.read_excel(RAW / "ine_empleo_provincial_76927.xlsx", header=None)
    header = raw.iloc[:8].ffill(axis=1)
    cols = [
        c for c in raw.columns
        if header[c].astype(str).map(snake).str.contains("empleo_total_miles_de_personas").any()
        and re.search(r"(?:19|20)\d{2}", str(raw.loc[7, c]))
    ]
    years = {c: int(re.search(r"(?:19|20)\d{2}", str(raw.loc[7, c])).group()) for c in cols}
    province_indices = [int(i) for i in raw.index if province_rows(pd.Series([raw.iloc[i, 0]])).iloc[0]]
    records = []
    for position, idx in enumerate(province_indices):
        end = province_indices[position + 1] if position + 1 < len(province_indices) else len(raw)
        block = raw.iloc[idx + 1:end]
        total_rows = block.index[block.iloc[:, 0].astype(str).map(snake).eq("total_personas")]
        if total_rows.empty:
            continue
        for col, year in years.items():
            records.append({
                "province_code": province_code(pd.Series([raw.iloc[idx, 0]])).iloc[0],
                "year": year,
                "empleo_total_miles_personas": numeric(pd.Series([raw.loc[int(total_rows[0]), col]])).iloc[0],
            })
    return expand_annual_component(
        pd.DataFrame(records), ["empleo_total_miles_personas"], 8
    )


def process_migration() -> pd.DataFrame:
    raw = pd.read_excel(RAW / "ine_saldos_migratorios_69765.xlsx", header=None)
    province_indices = [int(i) for i in raw.index if province_rows(pd.Series([raw.iloc[i, 0]])).iloc[0]]
    records = []
    for position, idx in enumerate(province_indices):
        end = province_indices[position + 1] if position + 1 < len(province_indices) else len(raw)
        block = raw.iloc[idx + 1:end]
        total_age = block.index[block.iloc[:, 0].astype(str).map(snake).eq("total")]
        if total_age.empty:
            continue
        for row_idx in range(int(total_age[0]) + 1, min(int(total_age[0]) + 8, end)):
            label = str(raw.iloc[row_idx, 0]).strip()
            if not re.fullmatch(r"(?:19|20)\d{2}", label):
                continue
            records.append({
                "province_code": province_code(pd.Series([raw.iloc[idx, 0]])).iloc[0],
                "year": int(label),
                "saldo_migratorio_total": numeric(pd.Series([raw.iloc[row_idx, 1]])).iloc[0],
                "saldo_migratorio_exterior": numeric(pd.Series([raw.iloc[row_idx, 2]])).iloc[0],
                "saldo_migratorio_interprovincial": numeric(pd.Series([raw.iloc[row_idx, 3]])).iloc[0],
            })
    return expand_annual_component(
        pd.DataFrame(records),
        ["saldo_migratorio_total", "saldo_migratorio_exterior", "saldo_migratorio_interprovincial"],
        8,
    )

def merge_components(components: Iterable[pd.DataFrame]) -> pd.DataFrame:
    frames = list(components)
    return reduce(
        lambda left, right: left.merge(
            right, on=["province_code", "quarter"], how="left", validate="one_to_one"
        ),
        frames,
    )


def add_features(panel: pd.DataFrame) -> pd.DataFrame:
    panel = complete_quarterly_grid(panel)

    def safe_ratio(numerator: str, denominator: str, scale: float = 1.0) -> pd.Series:
        values = scale * panel[numerator] / panel[denominator].replace(0, np.nan)
        return values.replace([np.inf, -np.inf], np.nan)

    if "poblacion_extranjera" in panel and "poblacion" in panel:
        panel["pct_poblacion_extranjera"] = (
            100 * panel["poblacion_extranjera"] / panel["poblacion"]
        )

    if "poblacion" in panel:
        for col in ("poblacion_18_34", "poblacion_25_39"):
            if col in panel:
                panel[f"pct_{col}"] = 100 * panel[col] / panel["poblacion"]
                panel[f"crecimiento_{col}_yoy"] = (
                    (panel[col] / calendar_shift(panel, col, 4) - 1) * 100
                )

    for col in ("hipotecas_numero", "viajeros_hoteles", "pernoctaciones_hoteles"):
        if col in panel and "poblacion" in panel:
            panel[f"{col}_por_1000_hab"] = 1000 * panel[col] / panel["poblacion"]

    if {"hipotecas_importe_miles_eur", "hipotecas_numero"}.issubset(panel.columns):
        panel["importe_medio_hipoteca_eur"] = safe_ratio(
            "hipotecas_importe_miles_eur", "hipotecas_numero", 1000
        )
    if {"compraventas_viviendas_total", "poblacion"}.issubset(panel.columns):
        panel["compraventas_por_1000_hab"] = safe_ratio(
            "compraventas_viviendas_total", "poblacion", 1000
        )
    for col in ("iniciadas_libre", "iniciadas_protegida", "terminadas_libre", "terminadas_protegida"):
        if col in panel and "poblacion" in panel:
            panel[f"{col}_por_1000_hab"] = safe_ratio(col, "poblacion", 1000)
    if {"pib_miles_eur", "poblacion"}.issubset(panel.columns):
        panel["pib_per_capita_eur"] = safe_ratio("pib_miles_eur", "poblacion", 1000)
    if {"vab_construccion_miles_eur", "pib_miles_eur"}.issubset(panel.columns):
        panel["peso_construccion_pib_pct"] = safe_ratio(
            "vab_construccion_miles_eur", "pib_miles_eur", 100
        )
    if {"empleo_total_miles_personas", "poblacion"}.issubset(panel.columns):
        panel["empleo_sobre_poblacion_pct"] = safe_ratio(
            "empleo_total_miles_personas", "poblacion", 100_000
        )
    if {"valor_tasado_m2", "renta_neta_media_hogar"}.issubset(panel.columns):
        panel["precio_m2_sobre_renta_hogar_pct"] = safe_ratio(
            "valor_tasado_m2", "renta_neta_media_hogar", 100
        )
    if {"precio_suelo_m2", "valor_tasado_m2"}.issubset(panel.columns):
        panel["ratio_precio_suelo_vivienda"] = safe_ratio(
            "precio_suelo_m2", "valor_tasado_m2"
        )

    if "valor_tasado_m2" in panel:
        for lag in (1, 2, 4):
            panel[f"valor_tasado_m2_lag{lag}"] = calendar_shift(panel, "valor_tasado_m2", lag)
        panel["valor_tasado_m2_qoq"] = ((panel["valor_tasado_m2"] / panel["valor_tasado_m2_lag1"] - 1) * 100).replace(
            [np.inf, -np.inf], np.nan
        )
        panel["valor_tasado_m2_yoy"] = ((panel["valor_tasado_m2"] / panel["valor_tasado_m2_lag4"] - 1) * 100).replace(
            [np.inf, -np.inf], np.nan
        )
        for horizon in (1, 4, 8, 12):
            panel[f"target_m2_t{horizon}"] = calendar_shift(panel, "valor_tasado_m2", -horizon)

    return panel


def write_availability_audit(components: dict[str, pd.DataFrame], panel: pd.DataFrame) -> None:
    """Record the actual availability convention and fingerprint the raw snapshot."""
    annual = {"income", "gdp", "regional_employment", "migration"}
    mixed = {"population", "young_population"}
    monthly = {"construction", "cpi", "tourism", "mortgages", "euribor", "deed_sales"}
    records = []
    for name, component in components.items():
        if name == "target":
            rule, delay, frequency = "Observed target at publication-time origin t", 0, "quarterly"
        elif name in annual:
            rule, delay, frequency = "Reference year Q4 + 8 quarters; carry forward after release", 8, "annual"
        elif name in mixed:
            rule, delay, frequency = "Annual: year Q4 + 8; quarterly: t + 1; carry released stock forward", None, "annual_and_quarterly"
        else:
            rule, delay, frequency = "Reference quarter + 1 quarter; no filling", 1, "monthly" if name in monthly else "quarterly"
        for column in component.select_dtypes(include=np.number):
            available = component.loc[component[column].notna(), "quarter"]
            in_panel = panel[column].notna()
            records.append({
                "component": name, "variable": column, "reference_frequency": frequency,
                "assumed_delay_quarters": delay, "availability_rule": rule,
                "first_available_quarter": str(available.min()),
                "last_available_quarter": str(available.max()),
                "nonmissing_panel_rows": int(in_panel.sum()),
                "panel_coverage_pct": round(100 * in_panel.mean(), 3),
                "availability_is_assumed": True, "historical_vintages_reconstructed": False,
            })
    audit = pd.DataFrame(records)
    audit.to_csv(REPORTS / "source_availability_audit.csv", index=False)
    missing = panel.loc[panel["valor_tasado_m2"].isna(), ["province_code", "province", "quarter"]].copy()
    missing["quarter"] = missing["quarter"].astype(str)
    missing.to_csv(REPORTS / "missing_target_quarters.csv", index=False)
    hashes = {}
    for path in sorted(RAW.iterdir()):
        if path.is_file():
            digest = hashlib.sha256()
            with path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
            hashes[path.name] = {"sha256": digest.hexdigest(), "bytes": path.stat().st_size}
    manifest = {
        "schema_version": 2,
        "quarter_axis": "forecast information origin; target t assumed observed on publication",
        "availability_convention": "assumed conservative delays; not actual release-calendar reconstruction",
        "quarterly_predictor_delay": QUARTERLY_PUBLICATION_DELAY,
        "annual_predictor_delay_after_reference_year_q4": ANNUAL_PUBLICATION_DELAY,
        "mixed_population_rule": "released quarterly stocks override delayed annual stocks; only forward carry",
        "historical_vintages_reconstructed": False,
        "revision_risk": "Downloaded latest-vintage data may contain revisions unavailable at historical origins",
        "target_imputation": "none; missing target calendar quarters retained",
        "feature_policy": "derived ratios and changes computed after availability alignment",
        "target_horizons": [1, 4, 8, 12],
        "first_quarter": str(panel["quarter"].min()), "last_quarter": str(panel["quarter"].max()),
        "rows": len(panel), "columns": len(panel.columns),
        "territories": int(panel["province_code"].nunique()),
        "observed_target_rows": int(panel["valor_tasado_m2"].notna().sum()),
        "missing_target_rows": len(missing),
        "raw_snapshot_files": hashes,
    }
    (REPORTS / "data_build_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    PROCESSED.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)

    processors = [
        ("target", process_target),
        ("transactions", process_transactions),
        ("construction", process_construction),
        ("epa", process_epa),
        ("population", process_population),
        ("young_population", process_young_population),
        ("foreign_population", process_foreign_population),
        ("cpi", process_cpi),
        ("tourism", process_tourism),
        ("mortgages", process_mortgages),
        ("euribor", process_bde_euribor),
        ("land_price", process_land_price),
        ("protected_housing", process_protected_housing),
        ("deed_sales", process_deed_sales),
        ("income", process_income),
        ("gdp", process_gdp),
        ("regional_employment", process_regional_employment),
        ("migration", process_migration),
    ]

    components: dict[str, pd.DataFrame] = {}
    failures = []
    annual_components = {"income", "gdp", "regional_employment", "migration"}
    mixed_components = {"population", "young_population"}
    for name, processor in processors:
        try:
            component = processor()
            if name not in annual_components | mixed_components | {"target"}:
                component = delay_quarterly_component(component)
            keys = ["province_code", "quarter"] if "province_code" in component else ["quarter"]
            if component.empty or component[keys].isna().any().any() or component.duplicated(keys).any():
                raise ValueError("Empty component, missing keys or duplicate observations")
            if "province_code" in component and not component["province_code"].isin(province_reference()).all():
                raise ValueError("Component contains a non-provincial geographical code")
            values = component.select_dtypes(include=np.number)
            if values.empty or values.notna().sum().eq(0).any():
                raise ValueError("Component has no usable data in one or more numeric variables")
            components[name] = component
        except Exception as exc:
            failures.append((name, f"{type(exc).__name__}: {exc}"))
            print(f"[ERROR] No se pudo procesar {name}: {exc}")

    try:
        metadata = process_ign_metadata()
        if metadata["province_code"].nunique() != 52 or metadata.duplicated("province_code").any():
            raise ValueError("IGN metadata must contain the 52 unique territories")
    except Exception as exc:
        failures.append(("ign_metadata", f"{type(exc).__name__}: {exc}"))
    if failures:
        (REPORTS / "panel_build_failures.json").write_text(json.dumps(failures, ensure_ascii=False, indent=2), encoding="utf-8")
        raise RuntimeError(f"Panel not published: {len(failures)} required source(s) failed: {failures}")

    # Retain all calendar quarters, including quarters without an observed target.
    target = complete_quarterly_grid(components["target"])
    target["province"] = target["province_code"].map(province_reference())

    other_components = [
        df for df in components.values()
        if "valor_tasado_m2" not in df.columns and "province_code" in df.columns
    ]
    panel = merge_components([target] + other_components)

    # Componentes nacionales, como Euríbor, se unen solo por trimestre.
    national_components = [
        df for df in components.values()
        if "province_code" not in df.columns and "quarter" in df.columns
    ]
    for component in national_components:
        panel = panel.merge(component, on="quarter", how="left", validate="many_to_one")

    # Metadatos territoriales estáticos.
    panel = panel.merge(metadata, on="province_code", how="left", validate="many_to_one")
    panel["densidad_poblacion"] = panel["poblacion"] / panel["superficie_km2"]

    panel = add_features(panel)
    for name, component in components.items():
        save_component(component, name)
    write_availability_audit(components, panel)

    panel["quarter"] = panel["quarter"].astype(str)
    csv_path = PROCESSED / "panel_provincial_trimestral.csv"
    parquet_path = PROCESSED / "panel_provincial_trimestral.parquet"
    panel.to_csv(csv_path, index=False)
    panel.to_parquet(parquet_path, index=False)

    print(f"\nPanel: {panel.shape}")
    print(f"CSV: {csv_path}")
    print(f"Parquet: {parquet_path}")

if __name__ == "__main__":
    main()
