#  Informe Comparativo de Scripts de Análisis Exploratorio (EDA)

Este documento presenta una comparación entre dos scripts desarrollados para la fase de *Data Understanding* del ciclo CRISP-DM. Ambos permiten cargar, procesar y analizar datos desde archivos Excel o CSV, pero con enfoques y alcances diferentes.

---

## 1. Descripción General

El primer script (`DUL2`) es una herramienta autónoma, ejecutable desde la línea de comandos, enfocada en análisis exploratorio, anonimización y visualización temática. Genera reportes gráficos por variable y cruces clave, ideal para entregas automáticas y seguras.

El segundo script (`CLASS_DATA_UNDERSTANDING`) sigue un enfoque modular, dividido en varios archivos (`src/utils`, `src/data`, `src/eda`), con buenas prácticas como uso de logs y estructuras limpias, pensado para desarrollo en entornos como VS Code.

---

## 2. Seguridad y Protección de Datos

Una diferencia clave es el manejo de datos sensibles:

- El primer script elimina columnas sensibles como nombres, direcciones, correos y documentos.
- Aplica enmascaramiento a datos personales en texto libre (como correos y teléfonos).
- Hashea valores de columnas sensibles, como el nombre del encuestador.

El segundo script no realiza ninguna de estas acciones, lo que lo hace menos adecuado para datos reales sin preprocesamiento adicional.

---

## 3. Análisis Temático y Cruces

El primer script organiza el análisis en **grupos temáticos**, como:

- Demografía
- Acceso territorial
- Motivos de gestión
- Experiencia y satisfacción

Además, genera análisis cruzados entre variables clave (por ejemplo: *Municipio × Punto de Atención*, *Motivo × Servicio*, *Calificación × Empresa*), lo que enriquece considerablemente la exploración.

El script modular no incluye esta funcionalidad.

---

## 4. Visualizaciones y Decisión Técnica

Ambos scripts generan gráficos de soporte, pero se identificó una **limitación crítica en el segundo script**:  
> La utilidad del análisis exploratorio se ve comprometida cuando las visualizaciones no son claras o no logran comunicar efectivamente los patrones y características de los datos.

Por esta razón, **se tomó la decisión de continuar el trabajo con el primer script**, el cual:

- Mejora la claridad visual de los gráficos.
- Organiza resultados por grupos temáticos.
- Incluye cruces clave entre variables.
- Automatiza la exportación de figuras útiles para la toma de decisiones.

---

## 5. Archivos de Salida

El script DUL2 genera:

- Diccionario de datos.
- Flags de calidad (columnas constantes, duplicadas, cardinalidad alta).
- Histogramas y gráficas categóricas por variable.
- Gráficos de cruces clave (Top-N pares).
- Datos intermedios seguros (`sample_head.csv`).
- Carpeta organizada de reportes y figuras.

El script class_data_understanding solo exporta algunas estadísticas básicas y gráficos generales, sin segmentación temática ni estructura clara de salida.

---

## 6. Conclusión

Ambos scripts son útiles, pero **el script `DUL2` es más robusto para entregas analíticas**, especialmente cuando se requiere:

- Seguridad y anonimización de datos.
- Visualizaciones más claras y relevantes.
- Análisis temático y segmentado.
- Automatización lista para producción.

El segundo script es más útil en contextos de desarrollo técnico, pruebas o mantenimiento modular, pero menos adecuado para generar reportes interpretables directamente.

---







