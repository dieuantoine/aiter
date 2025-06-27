import streamlit as st
import pandas as pd
import os, sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from config import DATA_DIR

# Charger le fichier CSV
df = pd.read_csv(DATA_DIR / "results_hyp_1.csv")

# Filtrer les colonnes à afficher
colonnes_affichees = [
    "request",
    "reference",
    "response",
    "reformulation",
    "score"
]

# Vérifier que les colonnes existent (au cas où)
colonnes_existantes = [col for col in colonnes_affichees if col in df.columns]

st.title("Tableau des Résultats de Reformulation")

# Affichage du DataFrame filtré
st.dataframe(df[colonnes_existantes], use_container_width=True)