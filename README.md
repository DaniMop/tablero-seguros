# Tablero de cartera de autos

Tablero en Streamlit sobre la cartera de autos del Módulo 4 (Diplomado ML en Seguros, FC UNAM):
filtros por cobertura, sexo, uso, combustible y edad; KPIs de riesgo (exposición, siniestros,
frecuencia, severidad, prima pura) y rentabilidad (loss y combined ratio); gráficas por factor;
y un cotizador con tres primas en orden: empírica, GLM y prima de tarifa.

## Correr local

```
pip install -r requirements.txt
streamlit run app_streamlit.py
```

La base `cartera.db` no se incluye en el repo: la app la reconstruye sola desde
`datos/datos.parquet` la primera vez que corre.
