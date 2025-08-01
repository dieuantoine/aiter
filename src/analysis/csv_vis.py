import streamlit as st
import pandas as pd
import os, sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from config import RESULTS_DIR, RESULTS_ANALYSIS_DIR

df = pd.read_csv(RESULTS_ANALYSIS_DIR / "aggregated_results.csv")

colonnes_affichees = [
    "model_id",
    "request",
    "reference",
    "hypothesis",
    "filtered_hypothesis",
    "corrected_hypothesis",
    "score",
    "ot_score",
    "global_score"
]

colonnes_existantes = [col for col in colonnes_affichees if col in df.columns]

st.title("Tableau des Résultats")

# st.dataframe(df[colonnes_existantes], use_container_width=True)

st.dataframe(df, use_container_width=True)