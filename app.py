"""Aplicación Streamlit para clasificar conexiones de red con el SVM RBF.

Ejecutar localmente:  streamlit run app.py
"""
import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

BASE = Path(__file__).resolve().parent
METADATOS = json.loads((BASE / "metadatos.json").read_text(encoding="utf-8"))
VARIABLES = METADATOS["variables_entrada"]
CLASES = {int(k): v for k, v in METADATOS["clases"].items()}
RANGO = METADATOS["rango_entrenamiento"]


@st.cache_resource
def cargar_modelo():
    return joblib.load(BASE / METADATOS["archivo_modelo"])


modelo = cargar_modelo()

st.set_page_config(page_title="Clasificador de conexiones", page_icon="🛡️")
st.title("🛡️ Clasificador de conexiones de red")
st.subheader("Presentado por Aaron Isaac Lopez 20211900269")
st.write(
    "Predice si una conexión es **normal** o **sospechosa** a partir del tráfico (Mbps) "
    "y la latencia (ms), usando un SVM con kernel RBF."
)

st.subheader("Predicción individual")
col1, col2 = st.columns(2)
trafico = col1.number_input(
    "Tráfico (Mbps)", min_value=0.0, step=1.0,
    value=float(round((RANGO["trafico_mbps"]["min"] + RANGO["trafico_mbps"]["max"]) / 2, 2)),
)
latencia = col2.number_input(
    "Latencia (ms)", min_value=0.0, step=1.0,
    value=float(round((RANGO["latencia_ms"]["min"] + RANGO["latencia_ms"]["max"]) / 2, 2)),
)

if st.button("Clasificar conexión", type="primary"):
    entrada = pd.DataFrame([[trafico, latencia]], columns=VARIABLES)
    pred = int(modelo.predict(entrada)[0])
    prob = float(modelo.predict_proba(entrada)[0][1])

    if pred == 1:
        st.error(f"Conexión **{CLASES[pred]}** · probabilidad de ser sospechosa: {prob:.1%}")
    else:
        st.success(f"Conexión **{CLASES[pred]}** · probabilidad de ser sospechosa: {prob:.1%}")

    fuera = [v for v, x in zip(VARIABLES, [trafico, latencia])
             if not RANGO[v]["min"] <= x <= RANGO[v]["max"]]
    if fuera:
        st.warning("Valores fuera del rango visto en entrenamiento para: "
                   + ", ".join(fuera) + ". La predicción puede ser poco fiable.")
