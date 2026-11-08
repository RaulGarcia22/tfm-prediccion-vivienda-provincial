"""Plantillas explicativas; no modifican la selección ni ajustan modelos."""
from textwrap import dedent


def md(text):
    return ('markdown', dedent(text).strip())


def code(text):
    return ('code', dedent(text).strip())


def quality_notebook(setup):
    return [
        md('''
        # 01 · Preparación y calidad de los datos

        **Objetivo:** caracterizar la variable de predicción y comprobar la calidad y disponibilidad de los datos utilizados en la evaluación.

        Este cuaderno desarrolla la trazabilidad, la unidad de análisis, el calendario, la disponibilidad y el tratamiento de ausencias. No entrena modelos ni elimina observaciones para mejorar resultados. Las transformaciones se ejecutan en los scripts 01–04; aquí se inspeccionan y explican.

        **Recorrido:** fuentes → panel → calendario → disponibilidad → cobertura → transformaciones → objetivos → extremos → límites. Las tablas se calculan al ejecutar, por lo que reflejan los archivos vigentes. Las inspecciones de toda la muestra se identifican como descriptivas; no sirven para retocar la selección congelada.
        '''),
        code(setup),
        code('''
        import numpy as np
        import sys
        if str(ROOT) not in sys.path:
            sys.path.insert(0, str(ROOT))
        from forecasting.direct import build_design
        from forecasting.experiment import SELECTION_CUTOFF, CALIBRATION_CUTOFF
        panel = pd.read_parquet(ROOT / 'data/processed/panel_provincial_trimestral.parquet')
        panel['quarter'] = pd.PeriodIndex(panel.quarter, freq='Q')
        panel['province_code'] = panel.province_code.astype(str).str.zfill(2)
        manifest = json.loads((ROOT / 'data/reports/data_build_manifest.json').read_text(encoding='utf-8'))
        coverage = pd.read_csv(EDA / 'cobertura_variables.csv')
        calendar = pd.read_csv(EDA / 'auditoria_calendario_provincial.csv', dtype={'province_code': str})
        '''),
        md('''
        ## 1. Qué representa cada observación

        Una fila corresponde a un territorio y un trimestre. El objetivo es el valor medio **tasado de vivienda libre**, en euros nominales por metro cuadrado. No es el precio individual de una compraventa ni un índice corregido por las características de las viviendas. Cambios en la composición de las tasaciones pueden afectar a la media.

        Se incluyen 50 provincias y Ceuta y Melilla. Una media simple de estos territorios no es una estimación ponderada del precio nacional. La amplitud temporal del panel tampoco implica que todas las variables estén observadas desde el principio.
        '''),
        code('''
        summary = pd.Series({
            'Filas de calendario': len(panel), 'Territorios': panel.province_code.nunique(),
            'Trimestres': panel.quarter.nunique(), 'Columnas': panel.shape[1],
            'Inicio': str(panel.quarter.min()), 'Fin': str(panel.quarter.max()),
            'Precios observados': int(panel.valor_tasado_m2.notna().sum()),
            'Precios ausentes': int(panel.valor_tasado_m2.isna().sum()),
        }, name='Valor')
        display(summary.to_frame())
        display(panel[['province_code', 'province', 'quarter', 'valor_tasado_m2']].head(8))
        '''),
        md('''
        ## 2. Fuentes y trazabilidad

        MIVAU proporciona tasaciones y actividad inmobiliaria; el INE aporta indicadores económicos, sociales y demográficos; el Banco de España, tipos de interés; el IGN, metadatos territoriales. `sources.json` es el inventario de descarga, no una lista de predictores ganadores. Algunas fuentes registradas son opcionales o manuales.

        La tabla muestra si el archivo está en la instantánea local. El manifiesto conserva las huellas de esa instantánea: permiten detectar cambios, pero no reconstruyen las versiones que publicaban los organismos en cada fecha histórica.
        '''),
        code('''
        sources = json.loads((ROOT / 'sources.json').read_text(encoding='utf-8'))
        inventory = []
        for category, entries in sources.items():
            for name, item in entries.items():
                filename = item.get('filename')
                path = ROOT / 'data/raw' / filename if filename else None
                inventory.append({'fuente': name, 'grupo': category,
                    'frecuencia': item.get('frequency'), 'ámbito': item.get('geography'),
                    'archivo_local': bool(path and path.is_file()),
                    'MB': round(path.stat().st_size / 1e6, 2) if path and path.is_file() else np.nan})
        display(pd.DataFrame(inventory))
        display(pd.Series({k: manifest[k] for k in ['target_imputation', 'historical_vintages_reconstructed',
                                                   'availability_convention', 'revision_risk']}))
        '''),
        md('''
        ## 3. De frecuencias distintas a un panel trimestral

        Los códigos provinciales se normalizan antes de unir fuentes. Los flujos mensuales se suman por trimestre y las tasas o tipos se promedian. Los stocks y los datos anuales se mantienen desde su disponibilidad supuesta; no se interpolan utilizando una publicación futura. Después se construyen intensidades por habitante y retardos.

        Una suma trimestral requiere interpretar correctamente las unidades de la fuente. Que la clave sea única no demuestra por sí solo que estén presentes los tres meses originales: también deben revisarse las categorías y ausencias de los archivos de entrada. El proceso aborta si falla un adaptador requerido, en vez de publicar silenciosamente un panel incompleto.
        '''),
        code('''
        keys = ['province_code', 'quarter']
        assert not panel[keys].isna().any().any()
        assert not panel.duplicated(keys).any()
        numeric = panel.select_dtypes(include=np.number).to_numpy(dtype=float, na_value=np.nan)
        assert not np.isinf(numeric).any()
        observed = panel.valor_tasado_m2.dropna()
        assert observed.gt(0).all()
        print('Claves únicas, sin infinitos y precios observados positivos.')
        print('Estos controles estructurales no certifican por sí solos la exactitud de cada fuente.')
        '''),
        md('''
        ## 4. Calendario completo no significa precios completos

        Se conserva una rejilla trimestral completa. Si falta un precio, la fila permanece con un valor ausente. Esto evita confundir «la siguiente fila disponible» con «el siguiente trimestre».

        Distinguimos huecos internos de la serie y ausencias anteriores al primer precio o posteriores al último. Ninguno se convierte artificialmente en una etiqueta real de entrenamiento. El mapa siguiente representa **ausencias de precio**, no caídas del mercado.
        '''),
        code('''
        expected = pd.period_range(panel.quarter.min(), panel.quarter.max(), freq='Q')
        assert all(pd.PeriodIndex(g.quarter.sort_values()).equals(expected)
                   for _, g in panel.groupby('province_code'))
        display(calendar.loc[calendar.n_precios_observados.lt(calendar.n_filas)])
        internal = int(calendar.n_huecos_internos_precio.sum())
        print(f'Huecos internos de precio: {internal}; ausencias totales: {panel.valor_tasado_m2.isna().sum()}.')
        prices = panel.pivot(index='province', columns='quarter', values='valor_tasado_m2')
        incomplete = prices.loc[prices.isna().any(axis=1)]
        fig, ax = plt.subplots(figsize=(12, max(3, .35 * len(incomplete))))
        ax.imshow(incomplete.isna().to_numpy(), aspect='auto', cmap='Greys', vmin=0, vmax=1)
        ax.set_yticks(range(len(incomplete)), incomplete.index)
        ticks = np.arange(0, len(prices.columns), 12)
        ax.set_xticks(ticks, prices.columns[ticks].astype(str), rotation=45)
        ax.set(title='Ausencias de precio en toda la muestra: negro = ausente', xlabel='Trimestre')
        plt.tight_layout(); plt.show()
        '''),
        md('''
        ## 5. Ejemplo de retardo por calendario

        El ejemplo de Cuenca ilustra por qué la consulta debe hacerse por fecha. Si falta un trimestre intermedio, el precio de dos o tres trimestres antes no puede presentarse como `lag1`. Del mismo modo, el objetivo anual debe corresponder exactamente a t+4, aunque la serie tenga huecos. No se utiliza el relleno de precios para crear objetivos.
        '''),
        code('''
        example = panel.loc[panel.province_code.eq('16') & panel.quarter.between('2011Q4', '2013Q2'),
            ['quarter', 'valor_tasado_m2', 'valor_tasado_m2_lag1', 'target_m2_t4']].copy()
        example['fecha_del_lag1'] = example.quarter - 1
        example['fecha_del_objetivo'] = example.quarter + 4
        display(example)
        '''),
        md('''
        ## 6. Cuándo se considera conocida cada variable

        El origen se ancla en el último valor tasado considerado publicado. Para las variables mensuales o trimestrales externas se supone un trimestre de retraso. Para las anuales, ocho trimestres después del cuarto trimestre del año de referencia. La población combina datos anuales retrasados con stocks trimestrales publicados según la convención del panel.

        Estos supuestos son deliberadamente explícitos; no equivalen a un calendario real de publicaciones. Las revisiones retrospectivas siguen siendo una limitación. Los ratios se calculan después de alinear la disponibilidad de sus componentes.
        '''),
        code('''
        availability = pd.read_csv(ROOT / 'data/reports/source_availability_audit.csv')
        display(availability[['component', 'variable', 'reference_frequency', 'availability_rule',
                              'first_available_quarter', 'panel_coverage_pct']])
        timing = pd.DataFrame([
            {'caso': 'Indicador trimestral de 2025Q4', 'referencia': '2025Q4',
             'disponibilidad_supuesta': str(pd.Period('2025Q4') + 1)},
            {'caso': 'Indicador anual de 2020', 'referencia': '2020Q4',
             'disponibilidad_supuesta': str(pd.Period('2020Q4') + 8)},
        ])
        display(timing)
        '''),
        md('''
        ## 7. Cobertura: qué información puede aprender un modelo

        Una variable con buena cobertura actual puede no tener datos en los orígenes de entrenamiento. La comparación distingue toda la muestra, desarrollo hasta 2018Q4 y un corte inicial hasta 2010Q4. Este último es un diagnóstico de cobertura, no una máscara universal para todos los ajustes.

        No se debe atribuir a una variable información que no estaba disponible. Las columnas completamente vacías se excluyen usando solo el entrenamiento de cada ajuste; las ausencias restantes se imputan dentro del modelo.
        '''),
        code('''
        variables = ['ipc_yoy', 'euribor_12m', 'tasa_paro', 'renta_neta_media_persona',
                     'pib_per_capita_eur', 'hipotecas_numero_por_1000_hab',
                     'viajeros_hoteles_por_1000_hab', 'pct_poblacion_18_34',
                     'poblacion_extranjera', 'saldo_migratorio_total']
        selected = coverage.set_index('variable').reindex(variables)
        display(selected[['primera_observacion', 'cobertura_train_inicial_pct',
                          'cobertura_desarrollo_pct', 'cobertura_total_pct']])
        selected[['cobertura_train_inicial_pct', 'cobertura_desarrollo_pct', 'cobertura_total_pct']].plot.barh(
            figsize=(11, 6), xlim=(0, 100), xlabel='Observaciones disponibles (%)')
        plt.title('Cobertura por periodo: los tamaños de muestra son diferentes')
        plt.tight_layout(); plt.show()
        display(Image(filename=str(EDA / 'figuras/cobertura_anual.png')))
        '''),
        md('''
        ## 8. Unidades, ratios y escala provincial

        Los totales de hipotecas o turistas están relacionados con el tamaño provincial. Por eso se consideran intensidades por 1.000 habitantes. El importe hipotecario original está en miles de euros; el ticket medio debe convertirse a euros. Estos indicadores responden preguntas diferentes: intensidad de actividad frente a volumen total.

        El IPC se usa como variación interanual, no para deflactar el valor tasado. La renta es anual y no equivale a un salario mensual. Los ratios dependen de la calidad y antigüedad de ambos componentes; normalizar no elimina todos los efectos territoriales.
        '''),
        code('''
        definitions = pd.DataFrame([
            ['Turismo por 1.000 habitantes', '1.000 × viajeros trimestrales / población disponible'],
            ['Hipotecas por 1.000 habitantes', '1.000 × número de hipotecas / población disponible'],
            ['Importe medio de hipoteca (euros)', '1.000 × importe en miles de euros / número de hipotecas'],
            ['Densidad (hab./km²)', 'población disponible / superficie'],
        ], columns=['Indicador', 'Definición'])
        display(definitions)
        check = panel.dropna(subset=['viajeros_hoteles', 'poblacion', 'viajeros_hoteles_por_1000_hab'])
        np.testing.assert_allclose(check.viajeros_hoteles_por_1000_hab,
                                   1000 * check.viajeros_hoteles / check.poblacion)
        print('La intensidad turística coincide con sus componentes disponibles.')
        '''),
        md('''
        ## 9. Etiquetas futuras y separación temporal

        En un ejemplo originado en t se quiere predecir t+h. Entrenar en una fecha exige que la etiqueta t+h ya sea observable. La columna descriptiva `split` de las matrices no sustituye este control, porque el periodo admisible depende del horizonte.

        Se distinguen desarrollo hasta 2018Q4, calibración posterior con objetivos hasta 2022Q4 y evaluación con objetivos desde 2023Q1. La tabla siguiente cuenta ejemplos etiquetados de desarrollo; no es el número de casos de los cuatro orígenes concretos de selección.
        '''),
        code('''
        counts = []
        cutoff = pd.Period(SELECTION_CUTOFF)
        for h in [1, 4, 8, 12]:
            design = build_design(panel, h)
            valid = design.quarter.le(cutoff) & design.target_quarter.le(cutoff)
            valid &= design.current_price.gt(0) & design.target_price.gt(0)
            counts.append({'horizonte': h, 'ejemplos_desarrollo': int(valid.sum()),
                           'último_origen_etiquetable': str(cutoff - h),
                           'última_fecha_objetivo': str(design.loc[valid, 'target_quarter'].max())})
        display(pd.DataFrame(counts))
        print('Una etiqueta ausente nunca se completa con la mediana.')
        '''),
        md('''
        ## 10. Distribuciones y extremos: investigar antes de excluir

        Los extremos pueden indicar errores, pero también diferencias reales entre provincias, episodios de crisis o denominadores pequeños. El informe marca valores fuera de tres rangos intercuartílicos y no los elimina. El diagnóstico siguiente se restringe a desarrollo.

        Un valor extremo del precio agregado no prueba que una vivienda individual esté mal tasada. Tampoco conviene eliminar episodios difíciles solo porque perjudican el modelo: forman parte de la capacidad predictiva que se pretende evaluar.
        '''),
        code('''
        distributions = pd.read_csv(EDA / 'distribuciones.csv')
        display(distributions.loc[distributions.ambito.eq('desarrollo') &
            distributions.variable.isin(['valor_tasado_m2', 'valor_tasado_m2_yoy', 'ipc_yoy',
                                        'euribor_12m', 'viajeros_hoteles_por_1000_hab'])])
        dev = panel.loc[panel.quarter.le(cutoff)]
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        axes[0].hist(dev.valor_tasado_m2.dropna(), bins=40, color='#2878B5')
        axes[0].set(title='Nivel del valor tasado en desarrollo', xlabel='€/m²', ylabel='Filas')
        axes[1].hist(dev.valor_tasado_m2_yoy.dropna(), bins=40, color='#E07A3F')
        axes[1].set(title='Variación interanual en desarrollo', xlabel='Variación simple (%)')
        plt.tight_layout(); plt.show()
        '''),
        md('''
        ## 11. Conclusiones de calidad y enlace con el análisis

        La rejilla y las etiquetas utilizan fechas trimestrales exactas; las ausencias se conservan. Las coberturas y fechas de publicación supuestas limitan qué pueden aprender los modelos. La imputación de predictores pertenece al entrenamiento, no a la construcción de los objetivos ni al conjunto completo.

        Las limitaciones que permanecen son las revisiones de las fuentes, el uso de retrasos supuestos, la mezcla de coberturas y el carácter agregado del valor tasado. Los controles automáticos reducen errores de implementación, pero no convierten las fuentes en una reconstrucción perfecta del pasado.

        **Siguiente cuaderno:** [02 · Análisis de variables](02_analisis_de_variables.ipynb). Estudia si las variables observadas aportan asociaciones con crecimientos futuros y si esas asociaciones son estables. No basta con que una columna esté completa para que sea útil.
        '''),
    ]


def variables_notebook(setup):
    return [
        md('''
        # 02 · Análisis de variables y dinámica provincial

        **Objetivo:** estudiar la información adicional de los indicadores económicos, demográficos y turísticos respecto a la historia del precio.

        Se estudian evolución, heterogeneidad, asociaciones en niveles y crecimientos, retardos adicionales, redundancia, estabilidad e información mutua. Cada apartado presenta las evidencias y su interpretación. El cuaderno no cambia modelos, hiperparámetros ni la selección congelada.

        **Separación de usos:** la evolución de toda la muestra es descriptiva. Las correlaciones, transformaciones exploratorias y diagnósticos que motivan variables emplean solo desarrollo, con orígenes y etiquetas hasta 2018Q4. Consultar resultados recientes aquí no convierte la evaluación en un test ciego.
        '''),
        code(setup),
        code('''
        import numpy as np
        import sys
        if str(ROOT) not in sys.path:
            sys.path.insert(0, str(ROOT))
        from forecasting.direct import build_design, FEATURE_BLOCKS, EXTERNAL_BLOCKS
        from forecasting.experiment import SELECTION_CUTOFF
        panel = pd.read_parquet(ROOT / 'data/processed/panel_provincial_trimestral.parquet')
        panel['quarter'] = pd.PeriodIndex(panel.quarter, freq='Q')
        panel['province_code'] = panel.province_code.astype(str).str.zfill(2)
        cutoff = pd.Period(SELECTION_CUTOFF)
        dev = panel.loc[panel.quarter.le(cutoff)].copy()
        corr = pd.read_csv(EDA / 'correlaciones_crecimiento.csv')
        levels = pd.read_csv(EDA / 'correlaciones_nivel.csv')
        stability = pd.read_csv(EDA / 'correlaciones_estabilidad.csv')
        coverage = pd.read_csv(EDA / 'cobertura_variables.csv')
        external = sorted({v for block in EXTERNAL_BLOCKS.values() for v in block})
        design4 = build_design(dev, 4)
        pairs4 = design4.loc[design4.target_quarter.le(cutoff)
                            & design4.current_price.gt(0) & design4.target_price.gt(0)].copy()
        assert pairs4.target_quarter.max() <= cutoff
        '''),
        md('''
        ## 1. Evolución del nivel y del crecimiento: descripción de toda la muestra

        Un nivel alto no implica un crecimiento alto. La primera gráfica muestra ambas dimensiones y la dispersión provincial. Las medianas y percentiles son resúmenes de territorios sin ponderación por población o transacciones; no representan un índice nacional.

        El área entre percentiles territoriales describe heterogeneidad observada. **No es un intervalo predictivo.** Las etapas históricas sirven para contextualizar, no constituyen una prueba formal de cambio estructural.
        '''),
        code('''
        display(Image(filename=str(EDA / 'figuras/evolucion_precio_crecimiento.png')))
        evolution = pd.read_csv(EDA / 'evolucion_anual.csv')
        display(evolution.tail(12))
        print('Este apartado incluye datos posteriores a desarrollo y es exclusivamente descriptivo.')
        '''),
        md('''
        ## 2. Diferencias territoriales al mismo punto de partida

        Se compara el cierre de desarrollo, no el último dato de evaluación. El gráfico ordena territorios por valor tasado y la tabla muestra también su variación interanual. La identidad provincial puede explicar parte de las diferencias de nivel; eso no garantiza anticipar cambios temporales.

        Predecir correctamente que una provincia tiene precios más altos que otra puede producir un buen ajuste agregado y, a la vez, errores relevantes en el crecimiento de ambas.
        '''),
        code('''
        snapshot = dev.loc[dev.quarter.eq(cutoff), ['province', 'valor_tasado_m2', 'valor_tasado_m2_yoy']]
        snapshot = snapshot.dropna(subset=['valor_tasado_m2']).sort_values('valor_tasado_m2')
        display(pd.concat([snapshot.head(8), snapshot.tail(8)]))
        selected = pd.concat([snapshot.head(8), snapshot.tail(8)]).set_index('province')
        selected.valor_tasado_m2.plot.barh(figsize=(10, 6), color='#2878B5')
        plt.title(f'Ocho valores menores y mayores en {cutoff}: comparación territorial')
        plt.xlabel('Valor tasado (€/m²)'); plt.tight_layout(); plt.show()
        '''),
        md('''
        ## 3. Por qué se incluyen variables externas

        Los bloques representan dimensiones distintas: crédito y operaciones, economía, demografía, turismo y oferta. El IPC mide inflación general; no deflacta el objetivo. El euríbor representa condiciones financieras y pertenece al bloque de crédito. Los indicadores turísticos se normalizan por población para distinguir intensidad de tamaño territorial.

        Estas son motivaciones para contrastar predictores, no relaciones causales ya demostradas. La tabla indica las variables realmente incluidas en la parrilla; otras columnas del panel pueden quedar fuera por cobertura o diseño.
        '''),
        code('''
        display(pd.DataFrame([{'bloque': k, 'variables': ', '.join(v), 'n_externas': len(v)}
                              for k, v in EXTERNAL_BLOCKS.items()]))
        display(coverage.loc[coverage.variable.isin(external),
                            ['variable', 'primera_observacion', 'cobertura_desarrollo_pct',
                             'cobertura_train_inicial_pct']].sort_values('cobertura_desarrollo_pct'))
        '''),
        md('''
        ## 4. Correlación con el nivel frente a correlación con crecimiento futuro

        Spearman mide asociación monotónica, no mejora de un pronóstico. La correlación contemporánea con el nivel puede reflejar diferencias provinciales o tendencias compartidas. La asociación con `100 × log(P[t+4]/P[t])` permite estudiar la relación entre el indicador conocido en t y el crecimiento posterior.

        Comparar estas dos columnas no equivale a una prueba estadística de diferencia: las muestras disponibles pueden variar. Por eso se muestran los números de pares y no solo los coeficientes.
        '''),
        code('''
        level_table = levels.loc[levels.variable.isin(external), ['variable', 'n_pares', 'spearman_agrupada']]
        annual = corr.loc[corr.horizonte.eq(4) & corr.variable.isin(external)]
        comparison = level_table.merge(annual[['variable', 'n_pares', 'spearman_agrupada']],
            on='variable', suffixes=('_nivel', '_crecimiento_futuro'))
        display(comparison.sort_values('spearman_agrupada_crecimiento_futuro', key=abs, ascending=False))
        focus = ['ipc_yoy', 'euribor_12m', 'tasa_paro', 'pib_per_capita_eur',
                 'viajeros_hoteles_por_1000_hab', 'hipotecas_numero_por_1000_hab']
        comparison.set_index('variable').reindex(focus)[
            ['spearman_agrupada_nivel', 'spearman_agrupada_crecimiento_futuro']].plot.barh(figsize=(11, 5))
        plt.axvline(0, color='grey', lw=.8); plt.xlim(-1, 1)
        plt.title('Asociación con nivel y crecimiento anual futuro: desarrollo')
        plt.tight_layout(); plt.show()
        '''),
        md('''
        ## 5. Asociación agrupada y asociación tras centrar por provincia

        El informe resta las medias provinciales de X e Y en la muestra de pares y aplica Spearman. Esto permite comprobar cuánto cambia la asociación al retirar diferencias medias territoriales. No es una regresión causal de efectos fijos, no elimina tendencias comunes y no resuelve la dependencia entre observaciones.

        Una diferencia grande entre ambas asociaciones invita a investigar si el patrón es territorial o temporal; no decide automáticamente si se debe conservar la variable.
        '''),
        code('''
        centered = annual[['variable', 'n_pares', 'spearman_agrupada', 'spearman_sin_media_provincial']].copy()
        centered['cambio_al_centrar'] = centered.spearman_sin_media_provincial - centered.spearman_agrupada
        display(centered.sort_values('cambio_al_centrar', key=abs, ascending=False))
        display(Image(filename=str(EDA / 'figuras/correlaciones_crecimiento.png')))
        '''),
        md('''
        ## 6. IPC, euríbor, paro y turismo: inspección de las nubes de puntos

        Las nubes se construyen con indicadores del origen y etiquetas anuales ya observables dentro de desarrollo. El crecimiento se mide en puntos logarítmicos porcentuales, aproximadamente porcentajes para cambios pequeños. La línea horizontal solo marca crecimiento nulo.

        La dispersión permite detectar relaciones débiles, concentraciones y extremos que un coeficiente resume en exceso. No se dibuja una recta causal ni se interpretan las miles de filas como observaciones independientes.
        '''),
        code('''
        variables = ['ipc_yoy', 'euribor_12m', 'tasa_paro', 'viajeros_hoteles_por_1000_hab']
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        for ax, variable in zip(axes.flat, variables):
            sample = pairs4[[variable, 'target_log_growth']].dropna()
            ax.scatter(sample[variable], sample.target_log_growth, s=7, alpha=.18, rasterized=True)
            ax.axhline(0, color='grey', lw=.8)
            ax.set(title=f'{variable}: {len(sample)} pares', xlabel=variable,
                   ylabel='Crecimiento futuro a 4T (100 × log)')
        plt.tight_layout(); plt.show()
        '''),
        md('''
        ## 7. La asociación depende del horizonte

        Una variable puede estar asociada con el crecimiento del próximo trimestre y tener poca relación con el de tres años. Las etiquetas de todos los horizontes deben terminar antes del corte de desarrollo; por ello sus últimos orígenes y tamaños muestrales difieren.

        El mapa usa una escala fija de −1 a 1. No se escoge el horizonte de mayor correlación después de mirar el test: estos son diagnósticos exploratorios de desarrollo.
        '''),
        code('''
        focus = ['ipc_yoy', 'euribor_12m', 'tasa_paro', 'pib_per_capita_eur',
                 'renta_neta_media_persona', 'hipotecas_numero_por_1000_hab',
                 'viajeros_hoteles_por_1000_hab', 'pernoctaciones_hoteles_por_1000_hab',
                 'pct_poblacion_18_34', 'iniciadas_libre_por_1000_hab']
        by_horizon = corr.loc[corr.variable.isin(focus)].pivot(
            index='variable', columns='horizonte', values='spearman_sin_media_provincial').reindex(focus)
        display(by_horizon)
        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(by_horizon.to_numpy(), aspect='auto', cmap='RdBu_r', vmin=-1, vmax=1)
        ax.set_yticks(range(len(by_horizon)), by_horizon.index)
        ax.set_xticks(range(len(by_horizon.columns)), by_horizon.columns)
        ax.set(xlabel='Horizonte (trimestres)', title='Spearman tras centrar por provincia: desarrollo')
        fig.colorbar(im, ax=ax); plt.tight_layout(); plt.show()
        '''),
        md('''
        ## 8. Retardos adicionales de indicadores externos

        La disponibilidad ya está retrasada en el panel. El retardo adicional de 0, 1 o 4 trimestres se aplica **encima** de esa regla: no permite acceder a publicaciones futuras. Se consulta el trimestre exacto y los huecos permanecen ausentes.

        Este apartado comprueba si cambia la asociación al mirar información más antigua. No incorpora automáticamente esos retardos a los modelos seleccionados. Elegir el máximo de muchas correlaciones generaría optimismo y requeriría otra validación.
        '''),
        code('''
        lagged = pd.read_csv(EDA / 'correlaciones_retardos_externos.csv')
        lag_view = lagged.loc[lagged.horizonte.eq(4) & lagged.variable.isin(focus)]
        display(lag_view.pivot(index='variable', columns='retardo_adicional',
                               values='spearman_sin_media_provincial'))
        display(lag_view[['variable', 'retardo_adicional', 'n_pares']])
        '''),
        md('''
        ## 9. Estabilidad histórica: una media puede ocultar cambios

        Las ventanas 1995–2007, 2008–2013 y 2014–2018 son descriptivas y están prefijadas. Tanto origen como objetivo deben estar dentro de cada ventana. Se comparan signo, magnitud y tamaño de muestra.

        Un cambio de signo es una señal para investigar, no una prueba formal de ruptura. En particular, una celda vacía o con pocos pares no permite concluir ausencia de relación. Ninguna ventana de este análisis llega al periodo de evaluación reciente.
        '''),
        code('''
        stable = stability.loc[stability.horizonte.eq(4) & stability.variable.isin(focus)]
        signs = stable.pivot(index='variable', columns='regimen', values='spearman_sin_media_provincial')
        sizes = stable.pivot(index='variable', columns='regimen', values='n_pares')
        display(signs); display(sizes)
        flips = signs.apply(lambda row: bool((row.dropna() > 0).any() and (row.dropna() < 0).any()), axis=1)
        print('Variables con signos distintos entre ventanas disponibles:', ', '.join(flips.index[flips]) or 'ninguna')
        '''),
        md('''
        ## 10. Redundancia: añadir columnas puede repetir información

        Las pernoctaciones y los viajeros, o ciertas medidas de población, pueden compartir mucha información. La tabla de redundancia muestra pares con asociación elevada en desarrollo; el mapa compara un conjunto concreto de predictores sobre los datos disponibles.

        Se usan pares completos, por lo que no todos los coeficientes comparten la misma muestra. La regularización puede ayudar cuando existen predictores relacionados, pero no elimina el problema de cobertura ni garantiza una mejora fuera de muestra. No se eliminan columnas automáticamente por superar un umbral.
        '''),
        code('''
        redundancy = pd.read_csv(EDA / 'redundancia.csv')
        display(redundancy.loc[redundancy.variable_a.isin(external) & redundancy.variable_b.isin(external)])
        columns = ['ipc_yoy', 'euribor_12m', 'tasa_paro', 'pib_per_capita_eur',
                   'viajeros_hoteles_por_1000_hab', 'pernoctaciones_hoteles_por_1000_hab',
                   'hipotecas_numero_por_1000_hab', 'compraventas_por_1000_hab']
        matrix = dev[columns].corr(method='spearman', min_periods=40)
        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(matrix, cmap='RdBu_r', vmin=-1, vmax=1)
        ax.set_xticks(range(len(columns)), columns, rotation=65, ha='right')
        ax.set_yticks(range(len(columns)), columns)
        ax.set_title('Redundancia en desarrollo; observaciones disponibles por pares')
        fig.colorbar(im, ax=ax); plt.tight_layout(); plt.show()
        '''),
        md('''
        ## 11. Información mutua: otro diagnóstico, no un selector definitivo

        La información mutua busca dependencias que no tienen por qué ser monotónicas. Se estima univariadamente, dentro de desarrollo, con la semilla fijada y una imputación aprendida en las filas etiquetadas de ese ámbito. Se expresa en nats, no como porcentaje de precio explicado.

        Una puntuación alta puede reflejar patrones de ausencia o tiempo compartido, además de señal económica. La comparación con Spearman no decide qué modelo gana: ambos son análisis dentro de muestra y no miden directamente el error de predicción futuro.
        '''),
        code('''
        mi = pd.read_csv(EDA / 'informacion_mutua.csv')
        mi4 = mi.loc[mi.horizonte.eq(4) & mi.variable.isin(external)].copy()
        mi4['porcentaje_imputado'] = 100 * mi4.n_imputados / mi4.n_objetivos
        display(mi4.sort_values('informacion_mutua_nats', ascending=False)[
            ['variable', 'informacion_mutua_nats', 'n_observados', 'porcentaje_imputado']])
        top = mi4.nlargest(12, 'informacion_mutua_nats').sort_values('informacion_mutua_nats')
        top.set_index('variable').informacion_mutua_nats.plot.barh(figsize=(10, 5), color='#2878B5')
        plt.title('Información mutua con crecimiento anual: diagnóstico dentro de desarrollo')
        plt.xlabel('Nats'); plt.tight_layout(); plt.show()
        '''),
        md('''
        ## 12. Sensibilidad a la muestra disponible

        Dos correlaciones de variables distintas pueden haberse calculado sobre años y provincias diferentes. Aquí se repite la asociación anual de varios indicadores sobre una muestra común completa y se compara con la muestra disponible de cada uno. No se incluye renta en este ejemplo para evitar que su corta cobertura domine toda la comparación.

        La muestra común mejora la comparabilidad, pero reduce la información y puede seleccionar periodos particulares. No resuelve todos los sesgos de ausencia. El resultado no cambia los bloques del benchmark.
        '''),
        code('''
        variables = ['ipc_yoy', 'euribor_12m', 'tasa_paro', 'viajeros_hoteles_por_1000_hab',
                     'hipotecas_numero_por_1000_hab']
        common = pairs4.dropna(subset=variables + ['target_log_growth'])
        rows = []
        for variable in variables:
            available = pairs4[[variable, 'target_log_growth']].dropna()
            rows.append({'variable': variable, 'n_disponible': len(available), 'n_común': len(common),
                'spearman_disponible': available[variable].corr(available.target_log_growth, method='spearman'),
                'spearman_muestra_común': common[variable].corr(common.target_log_growth, method='spearman')})
        display(pd.DataFrame(rows))
        print('Orígenes de la muestra común:', common.quarter.min(), 'a', common.quarter.max())
        '''),
        md('''
        ## 13. Síntesis por variable y paso a la prueba predictiva

        La tabla combina cobertura, asociación anual centrada, cambios de signo históricos e información mutua. Permite formular hipótesis, pero no constituye un ranking automático definitivo. En particular, no prueba un efecto causal individual del IPC, de los tipos o del turismo.

        El siguiente cuaderno compara el error al añadir cada bloque con el obtenido usando solo la historia del precio, bajo las mismas condiciones temporales. Una asociación alta no garantiza una reducción del error; una variable económicamente relevante puede no aportar una señal adicional utilizable con los datos disponibles.
        '''),
        code('''
        synthesis = annual[['variable', 'n_pares', 'spearman_sin_media_provincial']].merge(
            coverage[['variable', 'cobertura_desarrollo_pct']], on='variable', how='left')
        synthesis = synthesis.merge(mi4[['variable', 'informacion_mutua_nats', 'porcentaje_imputado']],
                                    on='variable', how='left')
        synthesis['signos_distintos_entre_ventanas'] = synthesis.variable.map(flips)
        display(synthesis.sort_values('variable'))
        print('La inclusión efectiva se decide en selección temporal, no en esta tabla exploratoria.')
        '''),
        md('''
        ## 14. Conclusiones e interpretación del resultado del TFM

        El análisis distingue nivel y crecimiento, diferencias provinciales y temporales, disponibilidad y asociación. Los indicadores económicos se estudian y prueban, aunque no todos lleguen a la previsión final. Su exclusión no demuestra que carezcan de influencia económica; mide el resultado de configuraciones y muestras concretas.

        En el experimento guardado resultaron seleccionadas configuraciones con historia del precio y turismo o solo con historia del precio. Eso no convierte al turismo en una explicación causal ni garantiza una mejora en otra época. La evaluación posterior muestra que la selección no supera la tendencia interanual reciente; esa evidencia se desarrolla en el cuaderno 04, no se utiliza aquí para rediseñar retrospectivamente las variables.

        **Límites:** dependencia entre provincias y fechas, tendencias compartidas, coberturas desiguales, pocas fechas de selección y ausencia de versiones históricas de publicaciones. Una ampliación futura debe formular y evaluar nuevas hipótesis con separación temporal, en lugar de seleccionar las correlaciones más favorables de la evaluación conocida.

        **Siguiente cuaderno:** [03 · Selección de modelos y variables](03_seleccion_modelos_variables.ipynb).
        '''),
    ]
