"""Lectura verificable y presentación del experimento híbrido exploratorio."""
from __future__ import annotations

import json

import matplotlib.pyplot as plt
import pandas as pd

from forecasting.experiment import ROOT, REPORTS
from forecasting.stages import file_sha256
from .exploratory_notebooks import md, code

OUT = ROOT / 'data/reports/hybrid_experiment'
LABELS = {
    'selected': 'Selección original', 'persistence': 'Persistencia',
    'annual_trend': 'Tendencia interanual', 'holt': 'Holt',
    'damped_trend': 'Tendencia amortiguada', 'recursive_huber': 'Huber recursivo',
    'hybrid_annual_trend': 'Tendencia + selección', 'hybrid_holt': 'Holt + selección',
}


def read_hybrid_results():
    manifest = json.loads((OUT / 'manifest.json').read_text(encoding='utf-8'))
    selection = json.loads((REPORTS / 'selection.json').read_text(encoding='utf-8'))
    if manifest.get('status') != 'complete' or manifest['original_selection_id'] != selection['selection_id']:
        raise ValueError('Experimento híbrido incompleto o de otra selección; ejecutar 11.')
    if manifest['evaluation_manifest_sha256'] != file_sha256(REPORTS / 'evaluation_manifest.json'):
        raise ValueError('La evaluación original ha cambiado; repetir 11.')
    for name, expected in manifest['artifact_hashes'].items():
        if '/' in name or '\\' in name or file_sha256(OUT / name) != expected:
            raise ValueError('Artefacto híbrido modificado; repetir 11.')
    return {
        'weights': json.loads((OUT / 'frozen_weights.json').read_text(encoding='utf-8')),
        'grid': pd.read_csv(OUT / 'development_grid.csv'),
        'metrics': pd.read_csv(OUT / 'evaluation_metrics.csv'),
        'summary': pd.read_csv(OUT / 'summary.csv'),
    }


def plot_development(data):
    fig, ax = plt.subplots(figsize=(10, 4.5))
    for base, part in data['grid'].groupby('base'):
        part = part.sort_values('ml_weight')
        ax.plot(part.ml_weight, part.mean_relative_MAE, marker='o', label=LABELS[base])
        weight = data['weights']['weights'][base]
        score = part.loc[part.ml_weight.eq(weight), 'mean_relative_MAE'].iloc[0]
        ax.scatter([weight], [score], s=140, facecolors='none', edgecolors='black', zorder=5)
    ax.set(xlabel='Peso de la selección original (0 = referencia; 1 = selección)',
           ylabel='MAE relativo medio frente a persistencia',
           title='Peso elegido únicamente con objetivos de desarrollo hasta 2018Q4')
    ax.grid(alpha=.2); ax.legend(); fig.tight_layout()
    return fig


def plot_evaluation(data):
    pivot = data['metrics'].pivot(index='horizon', columns='role', values='MAE')
    roles = ['selected', 'annual_trend', 'holt', 'hybrid_annual_trend', 'hybrid_holt']
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for role in roles:
        axes[0].plot(pivot.index, pivot[role], marker='o', label=LABELS[role])
    axes[0].set(xlabel='Horizonte (trimestres)', ylabel='MAE (€/m²)', title='Evaluación retrospectiva reciente')
    axes[0].legend(fontsize=8)
    for role in ['hybrid_annual_trend', 'hybrid_holt']:
        axes[1].plot(pivot.index, 100 * (1 - pivot[role] / pivot.annual_trend),
                     marker='o', label=LABELS[role])
    axes[1].axhline(0, color='black', lw=1)
    axes[1].set(xlabel='Horizonte (trimestres)', ylabel='Reducción del MAE (%)',
                title='Ventaja frente a tendencia interanual: positivo = mejora')
    axes[1].legend(fontsize=8)
    for ax in axes:
        ax.set_xticks(pivot.index); ax.grid(alpha=.2)
    fig.tight_layout()
    return fig


def write_hybrid_report():
    data = read_hybrid_results()
    for name, make in [('development_weights.png', plot_development), ('evaluation_comparison.png', plot_evaluation)]:
        fig = make(data)
        fig.savefig(OUT / name, dpi=150, bbox_inches='tight')
        plt.close(fig)
    weights = data['weights']
    rows = ['| Método | MAE medio por horizonte (€/m²) | Gana a tendencia (de 12) | Gana a selección original (de 12) |',
            '| --- | ---: | ---: | ---: |']
    for row in data['summary'].itertuples():
        rows.append(f'| {LABELS[row.role]} | {row.mean_horizon_MAE:.2f} | {row.wins_vs_annual_trend} | {row.wins_vs_original_selected} |')
    pivot = data['metrics'].pivot(index='horizon', columns='role', values='MAE')
    preferred = 'hybrid_' + weights['preferred_base']
    winner = data['summary'].set_index('role').loc[preferred]
    descriptions = []
    for base, weight in weights['weights'].items():
        descriptions.append(f'- {LABELS[base]}: **{100*(1-weight):.0f} % referencia + {100*weight:.0f} % selección original**.')
    text = f'''# Experimento exploratorio: combinación de series temporales y ML

## Objetivo y diseño

Se evalúa si combinar una referencia temporal con los modelos seleccionados
reduce el error. Se prueban dos mezclas convexas en **escala de precios**:

`predicción = (1 − peso_ML) × referencia + peso_ML × selección_original`

La referencia es tendencia interanual o Holt amortiguado. El segundo componente
es el modelo ya elegido para cada horizonte (Huber, Ridge, regresión lineal o
Huber recursivo); no es una nueva búsqueda entre todos los modelos. Tampoco es
un modelo entrenado sobre residuos. Es integración explícita mediante combinación
ponderada de pronósticos. Conserva las variables de sus componentes originales.

Se prueban once pesos entre 0 y 1, en pasos de 0,1, para cada referencia: 22
configuraciones, incluidas referencias puras. Se ajusta **un único peso global por
referencia**, común a provincias y horizontes, para limitar grados de libertad.
Se minimiza la media entre los doce horizontes del MAE dividido por el MAE de
persistencia. Los empates favorecen menor peso de ML y después el nombre de base.
Los extremos permiten concluir que mezclar no aporta si el desarrollo los elige.

Se usan únicamente las fechas objetivo 2013Q4, 2015Q4, 2017Q4 y 2018Q4 para
elegir pesos y referencia preferida. Los componentes son predicciones históricas
fuera del entrenamiento de cada origen. Se reutilizan los pronósticos guardados:
la combinación aritmética no necesita reajustar modelos. Se verifican muestra
común, etiquetas, selección y manifiestos. El peso se guarda antes de leer las
predicciones de evaluación. No se usa calibración para elegirlo.

**Límite metodológico:** el experimento se propone después de conocer la evaluación
original. Sus resultados recientes son exploratorios y retrospectivos; no constituyen
una prueba confirmatoria nueva. Además, los pesos y los modelos originales usan las
mismas fechas de desarrollo: sus errores de desarrollo tienen optimismo de selección,
no son una validación anidada. Solo hay cuatro fechas objetivo de selección por horizonte.

## Pesos congelados

{chr(10).join(descriptions)}

Un peso ML de 1 significa que la selección descarta la referencia y reproduce
exactamente el modelo original; no debe presentarse como un híbrido efectivo.
Un peso de 0 descarta ML y conserva solo la referencia.

La combinación preferida **en desarrollo** es {LABELS[preferred]}. Esta elección
no se modifica para favorecer el resultado reciente. Los componentes siguen
reajustándose en cada origen conforme al experimento original; los pesos permanecen fijos.

Gráfico generado en local: `data/reports/hybrid_experiment/development_weights.png`.

## Resultados recientes

Objetivos desde 2023Q1 hasta 2026Q1, según horizonte. Cada método usa las mismas
provincias y orígenes dentro de cada horizonte. El promedio da el mismo peso a
cada horizonte; no es el error de una previsión nacional. Hay entre
{int(data['metrics'].origins.min())} y {int(data['metrics'].origins.max())} orígenes por horizonte.

{chr(10).join(rows)}

Gráfico generado en local: `data/reports/hybrid_experiment/evaluation_comparison.png`.

La opción preferida en desarrollo alcanza **{winner.mean_horizon_MAE:.2f} €/m²**
de MAE medio, frente a **{pivot.annual_trend.mean():.2f}** de tendencia interanual
y **{pivot.selected.mean():.2f}** de la selección original. Mejora a la tendencia
en **{int(winner.wins_vs_annual_trend)} de 12 horizontes**. A cuatro trimestres,
su MAE es **{pivot.loc[4, preferred]:.2f} €/m²**, frente a
**{pivot.loc[4, 'annual_trend']:.2f}** de tendencia y **{pivot.loc[4, 'selected']:.2f}**
de la selección original.

Mejorar al componente original no implica superar a la referencia sencilla ni
demuestra superioridad futura. Se informan las dos combinaciones, aunque una
obtenga mejores resultados recientes; no se elige retrospectivamente entre ellas.
`evaluation_balanced_metrics.csv` restringe todos los horizontes a los orígenes
comunes. `paired_uncertainty.csv` aporta intervalos descriptivos de diferencias de
error, remuestreando bloques de cuatro orígenes y manteniendo juntas las provincias;
con menos de ocho orígenes omite esos intervalos. No son garantías de significación
con datos independientes ni corrigen haber diseñado el experimento después de ver resultados.

No se han calibrado bandas específicas para las mezclas: las bandas originales
no pueden heredarse. Este experimento no demuestra mejoras en cobertura ni
cambia las previsiones finales publicadas hasta 2028.
Su incorporación como modelo final requeriría una nueva versión documentada
del protocolo, calibración propia y evaluación adicional sin usarla para ajustar.

## Reproducibilidad

Ejecutar desde la raíz, después del flujo original:

```powershell
.venv\\Scripts\\python.exe 11_experimento_hibrido.py
```

Los resultados están en `data/reports/hybrid_experiment/`. `protocol.json`
describe la búsqueda; `frozen_weights.json` fija decisiones y huellas;
`development_grid.csv` conserva toda la parrilla; `evaluation_predictions.parquet`
guarda los pronósticos por caso. `manifest.json` enlaza los resultados con la
selección y evaluación originales. El cuaderno 03 muestra los pesos de desarrollo
y el 04 la evaluación exploratoria. La extensión se ejecuta por separado del flujo
original para distinguir las conclusiones del estudio inicial de esta prueba posterior.
'''
    (ROOT / 'docs/experimento_hibrido.md').write_text(text, encoding='utf-8')


def hybrid_cells(stage):
    heading = ('## Extensión exploratoria: pesos de tendencia/Holt + ML' if stage == 'development'
               else '## Extensión exploratoria: evaluación de los híbridos')
    note = ('Pesos ajustados solo con desarrollo hasta 2018Q4. Los círculos marcan los pesos elegidos; '
            'el resultado reciente se muestra en el cuaderno 04.' if stage == 'development' else
            'Pesos congelados en desarrollo. Esta prueba se diseñó después de ver la evaluación original: '
            'no constituye una evaluación confirmatoria nueva. No se trasladan las bandas originales a las mezclas.')
    plot = 'plot_development' if stage == 'development' else 'plot_evaluation'
    output = "display(hybrid_data['weights']['weights'])" if stage == 'development' else "display(hybrid_data['summary'])"
    return [md(heading + '\n\n' + note + '\n\nDetalle: [experimento híbrido](../docs/experimento_hibrido.md).'),
            code(f'''
            import sys
            if str(ROOT) not in sys.path:
                sys.path.insert(0, str(ROOT))
            from reporting.hybrid_report import read_hybrid_results, {plot}
            if (ROOT / 'data/reports/hybrid_experiment/manifest.json').is_file():
                hybrid_data = read_hybrid_results()
                {output}
                {plot}(hybrid_data)
                plt.show()
            else:
                print('Ejecutar 11_experimento_hibrido.py para generar esta extensión opcional.')
            ''')]
