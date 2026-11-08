from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

ROOT = Path(__file__).resolve().parent
RAW_DIR = ROOT / "data" / "raw"
SOURCES = json.loads((ROOT / "sources.json").read_text(encoding="utf-8"))


def build_session() -> requests.Session:
    retry = Retry(
        total=5,
        connect=5,
        read=5,
        backoff_factor=1.0,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("GET",),
    )
    session = requests.Session()
    session.headers.update({
        "User-Agent": "TFM-panel-inmobiliario/1.0 (uso academico)",
        "Accept": "*/*",
    })
    session.mount("https://", HTTPAdapter(max_retries=retry))
    session.mount("http://", HTTPAdapter(max_retries=retry))
    return session


def download(session: requests.Session, item: dict[str, Any], overwrite: bool = False) -> Path:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    destination = RAW_DIR / item["filename"]

    if destination.exists() and destination.stat().st_size > 0 and not overwrite:
        print(f"[SKIP] {destination.name}")
        return destination

    print(f"[GET ] {item['url']}")
    with session.get(item["url"], timeout=120, stream=True) as response:
        response.raise_for_status()
        content_type = response.headers.get("content-type", "").lower()

        # Evita guardar una página de error como .xlsx o .csv.
        if "text/html" in content_type and item["format"] in {"csv", "xlsx"}:
            preview = response.text[:300]
            raise RuntimeError(
                f"El servidor devolvió HTML para {item['url']}.\nInicio: {preview}"
            )

        tmp = destination.with_suffix(destination.suffix + ".part")
        with tmp.open("wb") as fh:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    fh.write(chunk)
        tmp.replace(destination)

    print(f"[OK  ] {destination.name} ({destination.stat().st_size:,} bytes)")
    time.sleep(0.3)
    return destination


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--include-optional",
        action="store_true",
        help="Descarga también las fuentes opcionales.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Vuelve a descargar archivos existentes.",
    )
    args = parser.parse_args()

    groups = ["core"] + (["optional"] if args.include_optional else [])
    session = build_session()

    failures: list[tuple[str, str]] = []
    for group in groups:
        for name, item in SOURCES[group].items():
            try:
                download(session, item, overwrite=args.overwrite)
            except Exception as exc:
                failures.append((name, str(exc)))
                print(f"[FAIL] {name}: {exc}")

    if failures:
        print("\nDescargas fallidas:")
        for name, reason in failures:
            print(f"- {name}: {reason}")
        raise SystemExit(1)

    print("\nDescarga terminada.")


if __name__ == "__main__":
    main()
