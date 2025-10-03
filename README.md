# Analisis de resultados
## Data understanding

El script realiza un análisis de Data Understanding siguiendo estos pasos:

1) Carga de datos: Lee el archivo Excel o CSV
2) Normaliza nombres de columnas: Elimina espacios y los reemplaza por guiones bajos.
3) Guarda una copia de muestra: Exporta un CSV con las primeras filas y columnas normalizadas a sample_head.csv.
4) Resumen general: Calcula dimensiones (filas, columnas) y muestra las primeras columnas.
5) Diccionario de datos: Genera un diccionario de variables y lo guarda en data_dictionary.csv.
6) Flags de calidad: Calcula indicadores de calidad (nulos, duplicados, etc.) y los guarda en quality_flags.json

7) Gráficos:
- Barras de valores faltantes (missing_bar.png)
- Histogramas y boxplots de variables numéricas (figures)
- Matriz de correlación (corr_matrix.png)
- Tablas de perfilado:
- Resumen numérico (numeric_summary.csv)
- Frecuencias de las 20 categorías más comunes por variable categórica (reports/{col}_top20_value_counts.csv)
