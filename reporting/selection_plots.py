"""Gráficos interpretables de selección, sin utilizar la evaluación reciente."""
from .exploratory_notebooks import code, md


def selection_plot_cells():
    return [
        md('''
        ## Comparación visual de familias en el cribado

        Cada barra representa la **mejor configuración de una familia** según la media del MAE relativo a persistencia en los horizontes 1, 4, 8 y 12. Se conserva la misma configuración en esos cuatro horizontes: no se mezclan ganadores distintos para fabricar una barra más favorable.

        Un valor inferior a 1 mejora la persistencia. La escala relativa evita que los errores mayores de horizontes largos dominen por sus unidades. Es un resultado de selección, no una estimación independiente de rendimiento futuro. Las familias tienen distintas cantidades de configuraciones y no se ha realizado una búsqueda exhaustiva.
        '''),
        code('''
        import numpy as np
        import sys
        if str(ROOT) not in sys.path:
            sys.path.insert(0, str(ROOT))
        from forecasting.direct import FAMILY_LABELS
        from forecasting.unified import BASELINE_LABELS
        labels = {**FAMILY_LABELS, **BASELINE_LABELS}
        block_labels = {'ar': 'Solo precio', 'credit': 'Crédito', 'economy': 'Economía',
                        'demography': 'Demografía', 'tourism': 'Turismo', 'supply': 'Oferta', 'all': 'Todos'}
        ranking = pd.read_csv(REPORTS / 'screening_ranking.csv')
        best_family = ranking.sort_values(['relative_MAE', 'candidate_id']).drop_duplicates('family')
        best_family = best_family.sort_values('relative_MAE', ascending=False)
        fig, ax = plt.subplots(figsize=(10, 6))
        colors = np.where(best_family.relative_MAE < 1, '#2878B5', '#D57947')
        bars = ax.barh(best_family.family.map(labels), best_family.relative_MAE, color=colors)
        ax.bar_label(bars, fmt='%.3f', padding=4)
        ax.axvline(1, linestyle='--', color='black', lw=1, label='Persistencia = 1')
        ax.set(xlabel='MAE relativo medio en cuatro horizontes (menor es mejor)',
               title='Mejor configuración por familia: cribado histórico')
        ax.set_xlim(0, max(1.15, best_family.relative_MAE.max() * 1.18))
        ax.legend(); ax.grid(axis='x', alpha=.2)
        plt.tight_layout(); plt.show()
        display(best_family[['family', 'candidate_id', 'relative_MAE']])
        '''),
        md('''
        ## Qué bloques de variables reducen el error

        Los mapas comparan el mejor ajuste de cada bloque con el mejor ajuste autorregresivo de la misma familia y horizonte. **Verde significa mejora y rojo empeoramiento**; los números son porcentajes de reducción del MAE. Se utiliza una escala común en los tres paneles.

        Aquí se aprecia si economía, crédito, turismo, demografía u oferta aportan información adicional utilizable. Cambiar de bloque también puede cambiar los hiperparámetros óptimos: no es un efecto causal individual del IPC o del turismo. Que un bloque mejore una celda no garantiza que entre en la preselección global de configuraciones.
        '''),
        code('''
        families = ['ridge', 'elasticnet', 'huber']
        ordered_blocks = ['credit', 'economy', 'demography', 'tourism', 'supply', 'all']
        part = blocks.loc[blocks.family.isin(families) & blocks.feature_set.isin(ordered_blocks)]
        scale = max(1., float(part.improvement_vs_AR_pct.abs().max()))
        fig, axes = plt.subplots(1, 3, figsize=(15, 6), layout='constrained')
        for ax, family in zip(axes, families):
            matrix = part.loc[part.family.eq(family)].pivot(
                index='feature_set', columns='horizon', values='improvement_vs_AR_pct')
            matrix = matrix.reindex(index=ordered_blocks, columns=[1, 4, 8, 12])
            im = ax.imshow(matrix.to_numpy(), aspect='auto', cmap='RdYlGn', vmin=-scale, vmax=scale)
            ax.set_xticks(range(4), matrix.columns)
            ax.set_yticks(range(len(ordered_blocks)), [block_labels[b] for b in ordered_blocks])
            ax.set(title=labels[family], xlabel='Horizonte (trimestres)')
            for row in range(len(matrix)):
                for col in range(len(matrix.columns)):
                    value = matrix.iloc[row, col]
                    if pd.notna(value):
                        ax.text(col, row, f'{value:+.1f}', ha='center', va='center', fontsize=9,
                                color='white' if abs(value) > .7 * scale else 'black')
        fig.colorbar(im, ax=axes, label='Mejora del MAE frente a AR (%)', shrink=.85)
        fig.suptitle('Aportación de bloques externos en la selección histórica')
        plt.show()
        '''),
        md('''
        ## Selección congelada frente a las dos referencias principales

        Este gráfico muestra la reducción porcentual del MAE de la especificación elegida para cada horizonte respecto a persistencia y tendencia interanual. Se consulta `selected_models.csv`, no se vuelve a seleccionar utilizando resultados posteriores.

        Las barras positivas indican mejora. Las cifras están calculadas sobre los casos históricos de selección y, por construcción, son favorables a los ganadores. La comprobación fuera de esa selección pertenece al cuaderno 04; este gráfico no demuestra superioridad futura.
        '''),
        code('''
        winners = pd.read_csv(REPORTS / 'selected_models.csv').set_index('horizon').sort_index()
        comparisons = pd.DataFrame(index=winners.index)
        for family, label in [('persistence', 'Frente a persistencia'), ('annual_trend', 'Frente a tendencia interanual')]:
            reference = metrics.loc[metrics.family.eq(family)].set_index('horizon').MAE
            comparisons[label] = 100 * (1 - winners.MAE / reference.reindex(winners.index))
        ax = comparisons.plot.bar(figsize=(12, 5), color=['#2878B5', '#E07A3F'])
        ax.axhline(0, color='black', lw=.9)
        ax.set(title='Ventaja de la selección congelada dentro de desarrollo',
               xlabel='Horizonte (trimestres)', ylabel='Reducción del MAE (%)')
        ax.tick_params(axis='x', rotation=0); ax.grid(axis='y', alpha=.2)
        plt.tight_layout(); plt.show()
        display(winners[['family', 'feature_set', 'MAE']].join(comparisons))
        '''),
        md('''
        ## Sensibilidad al retirar un origen de selección

        Cada horizonte se evaluó en cuatro fechas objetivo. El diagnóstico retira un origen cada vez y comprueba si seguiría ganando la misma configuración entre las finalistas. Un 100 % significa coincidencia en las cuatro exclusiones; un 25 %, en una.

        Es un análisis **posterior y descriptivo**: no modifica el ganador ni añade nuevos modelos. Con tan pocos orígenes, tampoco equivale a una probabilidad de que el modelo sea el mejor ni garantiza estabilidad ante otro régimen económico.
        '''),
        code('''
        sensitivity = pd.read_csv(REPORTS / 'robustness_selection_stability.csv').sort_values('horizon')
        fig, ax = plt.subplots(figsize=(11, 4.5))
        bars = ax.bar(sensitivity.horizon, 100 * sensitivity.unchanged_fraction, color='#2878B5')
        ax.bar_label(bars, labels=[f'{int(round(p*n))}/{int(n)}'
                     for p, n in zip(sensitivity.unchanged_fraction, sensitivity.comparisons)], padding=4)
        ax.set(xticks=sensitivity.horizon, ylim=(0, 113),
               title='Cuántas exclusiones conservan la selección original',
               xlabel='Horizonte (trimestres)', ylabel='Coincidencia con la selección completa (%)')
        ax.grid(axis='y', alpha=.2); plt.tight_layout(); plt.show()
        print('Este diagnóstico no utiliza el periodo de evaluación reciente ni altera selection.json.')
        '''),
    ]
