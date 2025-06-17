import streamlit as st
import pandas as pd
from sacrebleu.metrics import TER

# Charger le fichier CSV
df = pd.read_csv("resultats_avec_ter.csv")

# Filtrer les colonnes à afficher
colonnes_affichees = [
    "question_content",
    "refers_to_model",
    "response_content",
    "reference",
    "reformulation",
    "TER"
]

# Vérifier que les colonnes existent (au cas où)
colonnes_existantes = [col for col in colonnes_affichees if col in df.columns]

st.title("Tableau des Résultats de Reformulation")

# Affichage du DataFrame filtré
st.dataframe(df[colonnes_existantes], use_container_width=True)