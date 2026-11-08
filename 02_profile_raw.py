from __future__ import annotations

import csv
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
RAW_DIR = ROOT / "data" / "raw"
REPORT_DIR = ROOT / "data" / "reports"
REPORT_PATH = REPORT_DIR / "raw_profile.md"


def read_csv_flexible(path: Path, nrows: int | None = None) -> pd.DataFrame:
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

    return pd.read_csv(
        path,
        sep=delimiter,
        encoding=encoding,
        nrows=nrows,
        low_memory=False,
    )


def profile_file(path: Path) -> str:
    lines = [f"## `{path.name}`", ""]
    try:
        if path.suffix.lower() == ".xlsx":
            xls = pd.ExcelFile(path)
            lines.append(f"- Hojas: {xls.sheet_names}")
            df = pd.read_excel(path, sheet_name=xls.sheet_names[0])
        elif path.suffix.lower() == ".csv":
            df = read_csv_flexible(path)
        else:
            lines.append("- Formato no procesado.")
            return "\n".join(lines)

        lines.extend([
            f"- Forma: `{df.shape}`",
            f"- Columnas: `{list(df.columns)}`",
            "",
            "Primeras filas:",
            "",
            "```text",
            df.head(5).to_string(index=False),
            "```",
            "",
        ])

        # Valores de muestra para columnas categóricas pequeñas.
        for col in df.columns[:20]:
            s = df[col].dropna()
            if s.empty:
                continue
            nunique = s.astype(str).nunique()
            if nunique <= 20:
                values = s.astype(str).drop_duplicates().head(20).tolist()
                lines.append(f"- `{col}` ({nunique} valores): {values}")
        lines.append("")
    except Exception as exc:
        lines.append(f"- ERROR: `{type(exc).__name__}: {exc}`")
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    files = sorted(p for p in RAW_DIR.glob("*") if p.is_file())
    if not files:
        raise SystemExit("No hay archivos en data/raw. Ejecuta 01_download_sources.py.")

    content = [
        "# Perfil de ficheros brutos",
        "",
        "Este informe sirve para comprobar los nombres reales de columnas y categorías",
        "antes de fijar los filtros de transformación.",
        "",
    ]
    for path in files:
        content.append(profile_file(path))

    REPORT_PATH.write_text("\n".join(content), encoding="utf-8")
    print(f"Informe creado: {REPORT_PATH}")


if __name__ == "__main__":
    main()
