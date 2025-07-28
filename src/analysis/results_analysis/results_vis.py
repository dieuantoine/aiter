import streamlit as st
import pandas as pd
import os, sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from config import DATA_DIR

df = pd.read_csv(DATA_DIR / "results/results_1.csv")

colonnes_affichees = [
    "model_id",
    "request",
    "reference",
    "response",
    "hypothesis",
    "filtered_hypothesis",
    "corrected_hypothesis",
    "score",
    "ot_score",
    "global_score"
]

colonnes_existantes = [col for col in colonnes_affichees if col in df.columns]

st.title("Tableau des Résultats")

st.dataframe(df[colonnes_existantes], use_container_width=True)