"""Reproduce el TFM completo a partir de las fuentes locales conservadas."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
STEPS = [
    ('data', '03_build_panel_skeleton.py'),
    ('matrices', '04_prepare_model_data.py'),
    ('eda', '04b_analisis_exploratorio.py'),
    ('selection', '05_benchmark_modelos.py'),
    ('calibration', '07_backtest_multihorizonte.py'),
    ('evaluation', '08_evaluar_candidatos_finales.py'),
    ('forecast', '09_generar_prevision_final.py'),
    ('robustness', '06_analisis_robustez.py'),
    ('documentation', '10_generar_documentacion.py'),
    ('tests', None),
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--from-step', choices=[step for step, _ in STEPS], default='data')
    parser.add_argument('--jobs', type=int, default=4)
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error('--jobs debe ser positivo')
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    environment = dict(os.environ, PYTHONUTF8='1', MPLBACKEND='Agg')
    report = ROOT / 'data/reports/pipeline_run.json'
    report.parent.mkdir(parents=True, exist_ok=True)
    record = {'started_utc': datetime.now(timezone.utc).isoformat(),
              'python': sys.version, 'from_step': args.from_step, 'steps': [],
              'status': 'running', 'raw_data_policy': 'local snapshot; no automatic redownload'}

    def save():
        report.write_text(json.dumps(record, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')

    save()
    start_index = [name for name, _ in STEPS].index(args.from_step)
    for name, filename in STEPS[start_index:]:
        command = [sys.executable, '-u', filename] if filename else [
            sys.executable, '-m', 'unittest', 'discover', '-s', 'tests', '-v']
        if name in {'selection', 'calibration', 'evaluation'}:
            command += ['--jobs', str(args.jobs)]
        print(f'\n[PIPELINE] {name}', flush=True)
        started = time.monotonic()
        result = subprocess.run(command, cwd=ROOT, env=environment, check=False)
        record['steps'].append({'step': name, 'exit_code': result.returncode,
                                'seconds': round(time.monotonic() - started, 2)})
        if result.returncode:
            record.update(status='failed', failed_step=name)
            save()
            raise SystemExit(result.returncode)
        save()
    record.update(status='complete', finished_utc=datetime.now(timezone.utc).isoformat())
    save()
    print(f'\n[COMPLETE] {report}', flush=True)


if __name__ == '__main__':
    main()
