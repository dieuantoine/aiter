from huggingface_hub import login
from datasets import load_dataset
import sys, os

import streamlit as st
import pandas as pd
import ast

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from config import HF_TOKEN, REQ_INFO_DS

@st.cache_data
def load_data():
    login(HF_TOKEN)
    df = load_dataset(REQ_INFO_DS)['train'].to_pandas()
    return df

df = load_data()

st.title("Exploration des Requêtes Utilisateurs")

all_thematics = sorted(set(t for sublist in df["categories"] for t in sublist))
all_mots = sorted(set(m for sublist in df["question_word"] for m in sublist))

selected_thematique = st.selectbox("Sélectionner une thématique", options=all_thematics)
selected_mot = st.selectbox("Sélectionner un mot interrogatif", options=all_mots)

filtered_df = df[
    df['categories'].apply(lambda lst: selected_thematique in lst) &
    df['question_word'].apply(lambda lst: selected_mot in lst)
]

items_per_page = 50
total_results = len(filtered_df)
total_pages = (total_results - 1) // items_per_page + 1

if total_results > 0:
    page = st.number_input("Page", min_value=1, max_value=total_pages, value=1, step=1)
    start_idx = (page - 1) * items_per_page
    end_idx = start_idx + items_per_page

    st.write(f"Résultats {start_idx + 1} à {min(end_idx, total_results)} sur {total_results}")
    st.dataframe(filtered_df.iloc[start_idx:end_idx])
else:
    st.warning("Aucune requête ne correspond à ces critères.")
