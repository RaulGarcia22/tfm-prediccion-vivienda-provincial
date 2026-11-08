"""Calibra bandas con modelos congelados y objetivos 2019-2022."""
from __future__ import annotations
import argparse
from forecasting.stages import run_calibration


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--jobs', type=int, default=4, help='Procesos de ajuste en paralelo.')
    args = parser.parse_args()
    run_calibration(jobs=args.jobs)


if __name__ == '__main__':
    main()
