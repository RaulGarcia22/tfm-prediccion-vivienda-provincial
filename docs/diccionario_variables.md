# Diccionario de datos y variables del experimento

## Fuentes

El inventario ejecutable es `sources.json`. Esta tabla distingue fuentes registradas de archivos presentes durante la ejecución local; los ficheros de datos no se distribuyen en GitHub. La tabla de disponibilidad identifica las variables incorporadas al panel. Una fuente manual u opcional registrada no acredita que se haya utilizado como predictor.

| Fuente | Descripción | Grupo | Archivo presente |
| --- | --- | --- | --- |
| [mivau_valor_tasado](https://cdn.mivau.gob.es/portal-web-mivau/Datos_MIVAU/CSV/VDP006_01.csv) | Target: valor medio tasado de vivienda por provincia y trimestre. | core | Sí |
| [mivau_transacciones](https://cdn.mivau.gob.es/portal-web-mivau/Datos_MIVAU/CSV/VDP003_01.csv) | Número y valor de transacciones inmobiliarias de vivienda. | core | Sí |
| [mivau_construccion](https://cdn.mivau.gob.es/portal-web-mivau/Datos_MIVAU/CSV/VDP005_01.csv) | Viviendas iniciadas y terminadas, libres y protegidas; el fichero abierto usa año y mes. | core | Sí |
| [ine_hipotecas](https://www.ine.es/wstempus/xlsx/es/DATOS_TABLA/76317) | Número e importe de hipotecas por provincia y mes. | core | Sí |
| [ine_epa_provincial](https://www.ine.es/wstempus/xlsx/es/DATOS_TABLA/65349) | Tasas de actividad, paro y empleo por provincia, sexo y trimestre. | core | Sí |
| [ine_poblacion](https://www.ine.es/wstempus/xlsx/es/DATOS_TABLA/59589) | Población residente por provincia, fecha, sexo y edad. | core | Sí |
| [ine_poblacion_extranjera](https://www.ine.es/wstempus/xlsx/es/DATOS_TABLA/59591) | Población residente española/extranjera por provincia y trimestre. | core | Sí |
| [ine_turismo_hoteles](https://www.ine.es/wstempus/xlsx/es/DATOS_TABLA/67190) | Viajeros y pernoctaciones hoteleras por provincia y mes. | core | Sí |
| [ine_ipc_provincial](https://www.ine.es/wstempus/xlsx/es/DATOS_TABLA/76143) | Tasas de variación del IPC general por provincia y mes. | core | Sí |
| [bde_tipos_interes](https://www.bde.es/webbe/es/estadisticas/compartido/datos/csv/cal_umv_mfi001.csv) | Euríbor, tipos oficiales del BCE y otros tipos mensuales. | core | Sí |
| [ign_provincias](https://www.ign.es/resources/ane/Informacion_Geografica_Destacada/IGN_INFOGEO_PROVINCIAS.xlsx) | Metadatos estáticos: código, superficie, capital, CCAA y costa. | core | Sí |
| [ine_compraventas](https://www.ine.es/wstempus/xlsx/es/DATOS_TABLA/6150) | Compraventas de vivienda nueva/usada y libre/protegida por provincia y mes. | optional | Sí |
| [ine_paro_juvenil_ccaa](https://www.ine.es/wstempus/xlsx/es/DATOS_TABLA/65334) | Paro por grupos de edad a nivel CCAA; puede asignarse a sus provincias, etiquetándolo como variable autonómica. | optional | Sí |
| [mivau_precio_suelo](https://cdn.mivau.gob.es/portal-web-mivau/Datos_MIVAU/CSV/VDP004_01.csv) | Precio medio del suelo por ámbito geográfico, tamaño municipal y trimestre. | optional | Sí |
| [mivau_vivienda_protegida_rehabilitacion](https://cdn.mivau.gob.es/portal-web-mivau/Datos_MIVAU/CSV/VDP007_01.csv) | Actuaciones de vivienda protegida y rehabilitación por provincia y mes. | optional | Sí |
| [ine_poblacion_historica](https://www.ine.es/wstempus/xlsx/es/DATOS_TABLA/2852) | Población provincial anual histórica del Padrón a 1 de enero. | optional | Sí |
| [ine_poblacion_edad_historica](https://www.ine.es/jaxi/files/_px/es/csv_bdsc/t20/e245/p08/03003.csv) | Población provincial anual por edad simple, nacionalidad y sexo desde 1998. | optional | Sí |
| [ine_censo_anual_edad](https://www.ine.es/wstempus/xlsx/es/DATOS_TABLA/68519) | Censo anual 2021-2025 por provincia, sexo, edad simple y nacionalidad. | optional | Sí |
| [ine_renta_hogares](https://www.ine.es/wstempus/xlsx/es/DATOS_TABLA/53689) | Renta neta media por persona y hogar por provincia. | optional | Sí |
| [ine_pib_provincial](https://www.ine.es/wstempus/xlsx/es/DATOS_TABLA/tpx/76926) | PIB y VAB por ramas de actividad y provincia. | optional | Sí |
| [ine_empleo_provincial](https://www.ine.es/wstempus/xlsx/es/DATOS_TABLA/tpx/76927) | Empleo total y asalariado por provincia y rama de actividad. | optional | Sí |
| [ine_saldos_migratorios](https://www.ine.es/wstempus/xlsx/es/DATOS_TABLA/69765) | Saldo migratorio exterior e interprovincial por provincia. | optional | Sí |
| [aeat_salarios_provinciales](https://sede.agenciatributaria.gob.es/Sede/estadisticas/mercado-trabajo-pensiones.html) | Salario medio anual, asalariados y percepciones por provincia. | manual | No |

## Variables del modelo directo

Todas las especificaciones directas incluyen el bloque AR y la provincia como categoría. Las columnas completamente vacías se excluyen dentro de cada entrenamiento. Las variables externas ya están alineadas por disponibilidad cuando el modelo las recibe.

| Variable | Definición |
| --- | --- |
| price_log | log(€/m²) |
| growth_1q | 100 × log(P[t]/P[t−1]) |
| growth_4q | 100 × log(P[t]/P[t−4]) |
| growth_lag1 | 100 × log(P[t−1]/P[t−2]) |
| growth_lag4 | 100 × log(P[t−4]/P[t−5]) |
| volatility_4q | Desviación típica de cuatro crecimientos log trimestrales |
| quarter_sin | adimensional |
| quarter_cos | adimensional |

`volatility_4q` usa desviación poblacional (`ddof=0`). Seno y coseno son funciones de `2π × número_trimestre / 4`. Los crecimientos logarítmicos del modelo no son idénticos a las variaciones porcentuales simples auxiliares del panel.

| Conjunto | Descripción | Variables numéricas | Externas añadidas |
| --- | --- | --- | --- |
| ar | Historia del precio | 8 | Ninguna |
| credit | Precio + crédito y operaciones | 12 | hipotecas_numero_por_1000_hab, importe_medio_hipoteca_eur, euribor_12m, compraventas_por_1000_hab |
| economy | Precio + economía | 14 | tasa_paro, ipc_yoy, renta_neta_media_persona, renta_neta_media_hogar, pib_per_capita_eur, empleo_sobre_poblacion_pct |
| demography | Precio + demografía | 13 | densidad_poblacion, pct_poblacion_18_34, pct_poblacion_25_39, crecimiento_poblacion_18_34_yoy, crecimiento_poblacion_25_39_yoy |
| tourism | Precio + turismo | 10 | viajeros_hoteles_por_1000_hab, pernoctaciones_hoteles_por_1000_hab |
| supply | Precio + oferta | 14 | iniciadas_libre_por_1000_hab, iniciadas_protegida_por_1000_hab, terminadas_libre_por_1000_hab, terminadas_protegida_por_1000_hab, precio_suelo_m2, peso_construccion_pib_pct |
| all | Todos los bloques | 31 | Unión de los cinco bloques |

La variable dependiente es `100 × log(P[t+h]/P[t])`, para h=1,…,12. Se devuelve a euros mediante `P[t] × exp(predicción/100)`. Los retardos y objetivos usan trimestres exactos: si falta t−1, no se sustituye por la última fila existente.

## Disponibilidad

Los datos anuales entran en el cuarto trimestre de su año de referencia más ocho trimestres y se arrastran únicamente hacia delante. Las fuentes mensuales/trimestrales externas entran un trimestre después de su periodo de referencia. Las series recientes de población trimestral sustituyen la anual retrasada cuando están disponibles. Los ratios se calculan después de ese alineamiento, de modo que pueden combinar magnitudes con distintas antigüedades informativas.

`source_availability_audit.csv` conserva el supuesto por componente. Sus extremos de disponibilidad describen el componente antes del recorte al panel, y pueden superar 2026Q1; no significan que el pronóstico haya leído ese futuro. La cobertura siguiente se calcula sobre las filas efectivas del panel.

## Panel completo

| Variable | Unidad o fórmula | Origen | Cobertura (%) | Uso | Disponibilidad |
| --- | --- | --- | --- | --- | --- |
| province_code | Código INE de dos caracteres | Derivada o identificador | 100.00 | Clave; provincia se codifica en modelos directos | No es una magnitud continua |
| quarter | Periodo trimestral | Derivada o identificador | 100.00 | Clave; provincia se codifica en modelos directos | No es una magnitud continua |
| province | Nombre territorial | Derivada o identificador | 100.00 | Auxiliar/descriptiva; fuera de la parrilla final | No confundir presencia en el panel con uso por el modelo |
| valor_tasado_m2 | €/m² nominales | target | 98.63 | Objetivo y base de transformaciones AR | Observed target at publication-time origin t |
| trans_numero_transacciones_nueva | Recuento del concepto de la fuente | transactions | 70.40 | Auxiliar/descriptiva; fuera de la parrilla final | Reference quarter + 1 quarter; no filling |
| trans_numero_transacciones_protegida_nueva | Recuento del concepto de la fuente | transactions | 70.40 | Auxiliar/descriptiva; fuera de la parrilla final | Reference quarter + 1 quarter; no filling |
| trans_numero_transacciones_protegida_segunda_mano | Recuento del concepto de la fuente | transactions | 70.40 | Auxiliar/descriptiva; fuera de la parrilla final | Reference quarter + 1 quarter; no filling |
| trans_numero_transacciones_segunda_mano | Recuento del concepto de la fuente | transactions | 70.40 | Auxiliar/descriptiva; fuera de la parrilla final | Reference quarter + 1 quarter; no filling |
| trans_valor_transacciones_nueva | Miles de euros; consultar concepto de fuente | transactions | 70.40 | Auxiliar/descriptiva; fuera de la parrilla final | Reference quarter + 1 quarter; no filling |
| trans_valor_transacciones_protegida_nueva | Miles de euros; consultar concepto de fuente | transactions | 70.40 | Auxiliar/descriptiva; fuera de la parrilla final | Reference quarter + 1 quarter; no filling |
| trans_valor_transacciones_protegida_segunda_mano | Miles de euros; consultar concepto de fuente | transactions | 70.40 | Auxiliar/descriptiva; fuera de la parrilla final | Reference quarter + 1 quarter; no filling |
| trans_valor_transacciones_segunda_mano | Miles de euros; consultar concepto de fuente | transactions | 70.40 | Auxiliar/descriptiva; fuera de la parrilla final | Reference quarter + 1 quarter; no filling |
| iniciadas_libre | Recuento del concepto de la fuente | construction | 57.60 | Auxiliar/descriptiva; fuera de la parrilla final | Reference quarter + 1 quarter; no filling |
| iniciadas_protegida | Recuento del concepto de la fuente | construction | 57.60 | Auxiliar/descriptiva; fuera de la parrilla final | Reference quarter + 1 quarter; no filling |
| terminadas_libre | Recuento del concepto de la fuente | construction | 57.60 | Auxiliar/descriptiva; fuera de la parrilla final | Reference quarter + 1 quarter; no filling |
| terminadas_protegida | Recuento del concepto de la fuente | construction | 57.60 | Auxiliar/descriptiva; fuera de la parrilla final | Reference quarter + 1 quarter; no filling |
| tasa_paro | % | epa | 76.80 | economy, all | Reference quarter + 1 quarter; no filling |
| poblacion | Personas | population | 81.60 | Auxiliar/descriptiva; fuera de la parrilla final | Annual: year Q4 + 8; quarterly: t + 1; carry released stock forward |
| poblacion_18_34 | Personas | young_population | 81.60 | Auxiliar/descriptiva; fuera de la parrilla final | Annual: year Q4 + 8; quarterly: t + 1; carry released stock forward |
| poblacion_25_39 | Personas | young_population | 81.60 | Auxiliar/descriptiva; fuera de la parrilla final | Annual: year Q4 + 8; quarterly: t + 1; carry released stock forward |
| poblacion_extranjera | Personas | foreign_population | 2.40 | Auxiliar/descriptiva; fuera de la parrilla final | Reference quarter + 1 quarter; no filling |
| ipc_yoy | % interanual | cpi | 100.00 | economy, all | Reference quarter + 1 quarter; no filling |
| viajeros_hoteles | Recuento del concepto de la fuente | tourism | 83.08 | Auxiliar/descriptiva; fuera de la parrilla final | Reference quarter + 1 quarter; no filling |
| pernoctaciones_hoteles | Recuento del concepto de la fuente | tourism | 83.08 | Auxiliar/descriptiva; fuera de la parrilla final | Reference quarter + 1 quarter; no filling |
| hipotecas_numero | Recuento del concepto de la fuente | mortgages | 73.60 | Auxiliar/descriptiva; fuera de la parrilla final | Reference quarter + 1 quarter; no filling |
| hipotecas_importe_miles_eur | Miles de euros; consultar concepto de fuente | mortgages | 73.60 | Auxiliar/descriptiva; fuera de la parrilla final | Reference quarter + 1 quarter; no filling |
| precio_suelo_m2 | €/m² de suelo | land_price | 64.71 | supply, all | Reference quarter + 1 quarter; no filling |
| vp_rehabilitacion_aprobaciones_definitivas_rehabilitacion | Recuento del concepto de la fuente | protected_housing | 48.00 | Auxiliar/descriptiva; fuera de la parrilla final | Reference quarter + 1 quarter; no filling |
| vp_rehabilitacion_aprobaciones_provisionales_rehabilitacion | Recuento del concepto de la fuente | protected_housing | 48.00 | Auxiliar/descriptiva; fuera de la parrilla final | Reference quarter + 1 quarter; no filling |
| vp_rehabilitacion_solicitudes_rehabilitacion_viviendas | Recuento del concepto de la fuente | protected_housing | 48.00 | Auxiliar/descriptiva; fuera de la parrilla final | Reference quarter + 1 quarter; no filling |
| vp_vivienda_certificacion_definitiva | Recuento del concepto de la fuente | protected_housing | 57.60 | Auxiliar/descriptiva; fuera de la parrilla final | Reference quarter + 1 quarter; no filling |
| vp_vivienda_certificacion_provisional | Recuento del concepto de la fuente | protected_housing | 57.60 | Auxiliar/descriptiva; fuera de la parrilla final | Reference quarter + 1 quarter; no filling |
| vp_vivienda_solicitudes_certificacion_viviendas | Recuento del concepto de la fuente | protected_housing | 57.60 | Auxiliar/descriptiva; fuera de la parrilla final | Reference quarter + 1 quarter; no filling |
| compraventas_vivienda_libre | Recuento del concepto de la fuente | deed_sales | 60.80 | Auxiliar/descriptiva; fuera de la parrilla final | Reference quarter + 1 quarter; no filling |
| compraventas_vivienda_nueva | Recuento del concepto de la fuente | deed_sales | 60.80 | Auxiliar/descriptiva; fuera de la parrilla final | Reference quarter + 1 quarter; no filling |
| compraventas_vivienda_protegida | Recuento del concepto de la fuente | deed_sales | 60.80 | Auxiliar/descriptiva; fuera de la parrilla final | Reference quarter + 1 quarter; no filling |
| compraventas_vivienda_usada | Recuento del concepto de la fuente | deed_sales | 60.80 | Auxiliar/descriptiva; fuera de la parrilla final | Reference quarter + 1 quarter; no filling |
| compraventas_viviendas_total | Recuento del concepto de la fuente | deed_sales | 60.80 | Auxiliar/descriptiva; fuera de la parrilla final | Reference quarter + 1 quarter; no filling |
| renta_neta_media_persona | Euros anuales | income | 26.15 | economy, all | Reference year Q4 + 8 quarters; carry forward after release |
| renta_neta_media_hogar | Euros anuales | income | 26.15 | economy, all | Reference year Q4 + 8 quarters; carry forward after release |
| pib_miles_eur | Miles de euros; consultar concepto de fuente | gdp | 75.20 | Auxiliar/descriptiva; fuera de la parrilla final | Reference year Q4 + 8 quarters; carry forward after release |
| vab_construccion_miles_eur | Miles de euros; consultar concepto de fuente | gdp | 75.20 | Auxiliar/descriptiva; fuera de la parrilla final | Reference year Q4 + 8 quarters; carry forward after release |
| empleo_total_miles_personas | Miles de personas | regional_employment | 75.20 | Auxiliar/descriptiva; fuera de la parrilla final | Reference year Q4 + 8 quarters; carry forward after release |
| saldo_migratorio_total | Recuento del concepto de la fuente | migration | 8.00 | Auxiliar/descriptiva; fuera de la parrilla final | Reference year Q4 + 8 quarters; carry forward after release |
| saldo_migratorio_exterior | Recuento del concepto de la fuente | migration | 8.00 | Auxiliar/descriptiva; fuera de la parrilla final | Reference year Q4 + 8 quarters; carry forward after release |
| saldo_migratorio_interprovincial | Recuento del concepto de la fuente | migration | 8.00 | Auxiliar/descriptiva; fuera de la parrilla final | Reference year Q4 + 8 quarters; carry forward after release |
| euribor_12m | % anual, promedio trimestral | euribor | 86.40 | credit, all | Reference quarter + 1 quarter; no filling |
| province_ign | Texto/categoría; consultar fuente | Derivada o identificador | 100.00 | Auxiliar/descriptiva; fuera de la parrilla final | No confundir presencia en el panel con uso por el modelo |
| superficie_km2 | km² | Derivada o identificador | 100.00 | Auxiliar/descriptiva; fuera de la parrilla final | No confundir presencia en el panel con uso por el modelo |
| capital | Texto/categoría; consultar fuente | Derivada o identificador | 100.00 | Auxiliar/descriptiva; fuera de la parrilla final | No confundir presencia en el panel con uso por el modelo |
| comunidad_autonoma | Texto/categoría; consultar fuente | Derivada o identificador | 100.00 | Auxiliar/descriptiva; fuera de la parrilla final | No confundir presencia en el panel con uso por el modelo |
| longitud_costa_km | km | Derivada o identificador | 100.00 | Auxiliar/descriptiva; fuera de la parrilla final | No confundir presencia en el panel con uso por el modelo |
| densidad_poblacion | Habitantes / km² | Derivada o identificador | 81.60 | demography, all | Predictor de origen; la inclusión efectiva depende del entrenamiento |
| pct_poblacion_extranjera | % | Derivada o identificador | 2.40 | Auxiliar/descriptiva; fuera de la parrilla final | No confundir presencia en el panel con uso por el modelo |
| pct_poblacion_18_34 | % | Derivada o identificador | 81.60 | demography, all | Predictor de origen; la inclusión efectiva depende del entrenamiento |
| crecimiento_poblacion_18_34_yoy | % | Derivada o identificador | 78.40 | demography, all | Predictor de origen; la inclusión efectiva depende del entrenamiento |
| pct_poblacion_25_39 | % | Derivada o identificador | 81.60 | demography, all | Predictor de origen; la inclusión efectiva depende del entrenamiento |
| crecimiento_poblacion_25_39_yoy | % | Derivada o identificador | 78.40 | demography, all | Predictor de origen; la inclusión efectiva depende del entrenamiento |
| hipotecas_numero_por_1000_hab | Numerador disponible × 1000 / población disponible | Derivada o identificador | 73.60 | credit, all | Predictor de origen; la inclusión efectiva depende del entrenamiento |
| viajeros_hoteles_por_1000_hab | Numerador disponible × 1000 / población disponible | Derivada o identificador | 78.46 | tourism, all | Predictor de origen; la inclusión efectiva depende del entrenamiento |
| pernoctaciones_hoteles_por_1000_hab | Numerador disponible × 1000 / población disponible | Derivada o identificador | 78.46 | tourism, all | Predictor de origen; la inclusión efectiva depende del entrenamiento |
| importe_medio_hipoteca_eur | Importe total (miles €) × 1000 / número de hipotecas | Derivada o identificador | 73.60 | credit, all | Predictor de origen; la inclusión efectiva depende del entrenamiento |
| compraventas_por_1000_hab | Numerador disponible × 1000 / población disponible | Derivada o identificador | 60.80 | credit, all | Predictor de origen; la inclusión efectiva depende del entrenamiento |
| iniciadas_libre_por_1000_hab | Numerador disponible × 1000 / población disponible | Derivada o identificador | 57.60 | supply, all | Predictor de origen; la inclusión efectiva depende del entrenamiento |
| iniciadas_protegida_por_1000_hab | Numerador disponible × 1000 / población disponible | Derivada o identificador | 57.60 | supply, all | Predictor de origen; la inclusión efectiva depende del entrenamiento |
| terminadas_libre_por_1000_hab | Numerador disponible × 1000 / población disponible | Derivada o identificador | 57.60 | supply, all | Predictor de origen; la inclusión efectiva depende del entrenamiento |
| terminadas_protegida_por_1000_hab | Numerador disponible × 1000 / población disponible | Derivada o identificador | 57.60 | supply, all | Predictor de origen; la inclusión efectiva depende del entrenamiento |
| pib_per_capita_eur | PIB (miles €) × 1000 / población disponible | Derivada o identificador | 75.20 | economy, all | Predictor de origen; la inclusión efectiva depende del entrenamiento |
| peso_construccion_pib_pct | 100 × VAB construcción / PIB | Derivada o identificador | 75.20 | supply, all | Predictor de origen; la inclusión efectiva depende del entrenamiento |
| empleo_sobre_poblacion_pct | 100000 × empleo (miles) / población disponible | Derivada o identificador | 75.20 | economy, all | Predictor de origen; la inclusión efectiva depende del entrenamiento |
| precio_m2_sobre_renta_hogar_pct | Texto/categoría; consultar fuente | Derivada o identificador | 26.15 | Auxiliar/descriptiva; fuera de la parrilla final | No confundir presencia en el panel con uso por el modelo |
| ratio_precio_suelo_vivienda | Texto/categoría; consultar fuente | Derivada o identificador | 64.46 | Auxiliar/descriptiva; fuera de la parrilla final | No confundir presencia en el panel con uso por el modelo |
| valor_tasado_m2_lag1 | €/m² | Derivada o identificador | 97.83 | Auxiliar/descriptiva; fuera de la parrilla final | No confundir presencia en el panel con uso por el modelo |
| valor_tasado_m2_lag2 | €/m² | Derivada o identificador | 97.03 | Auxiliar/descriptiva; fuera de la parrilla final | No confundir presencia en el panel con uso por el modelo |
| valor_tasado_m2_lag4 | €/m² | Derivada o identificador | 95.43 | Auxiliar/descriptiva; fuera de la parrilla final | No confundir presencia en el panel con uso por el modelo |
| valor_tasado_m2_qoq | % | Derivada o identificador | 97.68 | Auxiliar/descriptiva; fuera de la parrilla final | No confundir presencia en el panel con uso por el modelo |
| valor_tasado_m2_yoy | % | Derivada o identificador | 95.18 | Auxiliar/descriptiva; fuera de la parrilla final | No confundir presencia en el panel con uso por el modelo |
| target_m2_t1 | €/m² | Derivada o identificador | 97.86 | Etiqueta auxiliar, nunca predictor | Precio en t+h por fecha exacta; sin imputación |
| target_m2_t4 | €/m² | Derivada o identificador | 95.55 | Etiqueta auxiliar, nunca predictor | Precio en t+h por fecha exacta; sin imputación |
| target_m2_t8 | €/m² | Derivada o identificador | 92.48 | Etiqueta auxiliar, nunca predictor | Precio en t+h por fecha exacta; sin imputación |
| target_m2_t12 | €/m² | Derivada o identificador | 89.40 | Etiqueta auxiliar, nunca predictor | Precio en t+h por fecha exacta; sin imputación |

## Cobertura de predictores de la parrilla

| Variable | Primera disponible | Desarrollo (%) | Hasta 2010Q4 (%) | Panel (%) |
| --- | --- | --- | --- | --- |
| compraventas_por_1000_hab | 2007Q2 | 48.96 | 23.44 | 60.80 |
| crecimiento_poblacion_18_34_yoy | 2001Q4 | 71.88 | 57.81 | 78.40 |
| crecimiento_poblacion_25_39_yoy | 2001Q4 | 71.88 | 57.81 | 78.40 |
| densidad_poblacion | 2000Q4 | 76.04 | 64.06 | 81.60 |
| empleo_sobre_poblacion_pct | 2002Q4 | 67.71 | 51.56 | 75.20 |
| euribor_12m | 1999Q2 | 82.29 | 73.44 | 86.40 |
| growth_1q | 1995Q2 | 96.98 | 96.27 | 97.68 |
| growth_4q | 1996Q1 | 93.73 | 91.59 | 95.18 |
| growth_lag1 | 1995Q3 | 95.93 | 94.71 | 96.88 |
| growth_lag4 | 1996Q2 | 92.81 | 90.02 | 94.48 |
| hipotecas_numero_por_1000_hab | 2003Q2 | 65.62 | 48.44 | 73.60 |
| importe_medio_hipoteca_eur | 2003Q2 | 65.62 | 48.44 | 73.60 |
| iniciadas_libre_por_1000_hab | 2008Q2 | 44.79 | 17.19 | 57.60 |
| iniciadas_protegida_por_1000_hab | 2008Q2 | 44.79 | 17.19 | 57.60 |
| ipc_yoy | 1995Q1 | 100.00 | 100.00 | 100.00 |
| pct_poblacion_18_34 | 2000Q4 | 76.04 | 64.06 | 81.60 |
| pct_poblacion_25_39 | 2000Q4 | 76.04 | 64.06 | 81.60 |
| pernoctaciones_hoteles_por_1000_hab | 2000Q4 | 73.12 | 61.60 | 78.46 |
| peso_construccion_pib_pct | 2002Q4 | 67.71 | 51.56 | 75.20 |
| pib_per_capita_eur | 2002Q4 | 67.71 | 51.56 | 75.20 |
| precio_suelo_m2 | 2004Q2 | 56.43 | 38.85 | 64.71 |
| price_log | 1995Q1 | 98.22 | 97.84 | 98.63 |
| quarter_cos | 1995Q1 | 100.00 | 100.00 | 100.00 |
| quarter_sin | 1995Q1 | 100.00 | 100.00 | 100.00 |
| renta_neta_media_hogar | 2017Q4 | 4.83 | 0.00 | 26.15 |
| renta_neta_media_persona | 2017Q4 | 4.83 | 0.00 | 26.15 |
| tasa_paro | 2002Q2 | 69.79 | 54.69 | 76.80 |
| terminadas_libre_por_1000_hab | 2008Q2 | 44.79 | 17.19 | 57.60 |
| terminadas_protegida_por_1000_hab | 2008Q2 | 44.79 | 17.19 | 57.60 |
| viajeros_hoteles_por_1000_hab | 2000Q4 | 73.12 | 61.60 | 78.46 |
| volatility_4q | 1996Q1 | 93.27 | 91.59 | 94.83 |

Las tasas de cobertura distintas implican que las correlaciones pueden utilizar muestras diferentes. La imputación permite ajustar modelos, pero no crea evidencia observada donde una fuente todavía no existía.
