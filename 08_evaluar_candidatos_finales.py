"""Evalua la seleccion congelada en objetivos desde 2023, sin volver a seleccionar."""
from __future__ import annotations
import argparse
from forecasting.stages import run_evaluation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--jobs', type=int, default=4, help='Procesos de ajuste en paralelo.')
    args = parser.parse_args()
    run_evaluation(jobs=args.jobs)


if __name__ == '__main__':
    main()
