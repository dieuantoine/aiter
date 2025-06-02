from huggingface_hub import login
from datasets import load_dataset
import sys, os

import streamlit as st
import pandas as pd
import ast

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from config import HF_TOKEN, REQ_INFO_DS, SELECTED_IDS_CSV

if "selected_conv_ids" not in st.session_state:
    if os.path.exists(SELECTED_IDS_CSV):
        st.session_state.selected_conv_ids = pd.read_csv(SELECTED_IDS_CSV)["conversation_pair_id"].tolist()
    else:
        st.session_state.selected_conv_ids = []

@st.cache_data
def load_data():
    login(HF_TOKEN)
    df = load_dataset(REQ_INFO_DS)['train'].to_pandas()
    return df

df = load_data()

st.title("Exploration des requêtes")

all_thematics = sorted(set(t for sublist in df["categories"] for t in sublist))
all_mots = sorted(set(m for sublist in df["question_word"] for m in sublist))

selected_thematiques = st.multiselect("Sélectionner une ou plusieurs thématiques", options=all_thematics)
selected_mots = st.multiselect("Sélectionner un ou plusieurs mots interrogatifs", options=all_mots)
search_text = st.text_input("Recherche par mot-clé (optionnelle)")

filtered_df = df

if selected_thematiques:
    filtered_df = filtered_df[filtered_df['categories'].apply(lambda lst: any(t in lst for t in selected_thematiques))]

if selected_mots:
    filtered_df = filtered_df[filtered_df['question_word'].apply(lambda lst: any(m in lst for m in selected_mots))]

if search_text:
    filtered_df = filtered_df[filtered_df['opening_msg'].str.contains(search_text, case=False, na=False)]

items_per_page = 50
total_results = len(filtered_df)
total_pages = (total_results - 1) // items_per_page + 1

if total_results > 0:
    page = st.number_input("Page", min_value=1, max_value=total_pages, value=1, step=1)
    start_idx = (page - 1) * items_per_page
    end_idx = start_idx + items_per_page

    st.write(f"Résultats {start_idx + 1} à {min(end_idx, total_results)} sur {total_results}")
    for idx, row in filtered_df.iloc[start_idx:end_idx].iterrows():
        cols = st.columns([8, 2])
        with cols[0]:
            st.markdown(f"**Requête :** {row['opening_msg']}  \n**Thématiques :** {', '.join(row['categories'])}  \n**Mot(s) :** {', '.join(row['question_word'])}")
        with cols[1]:
            if st.button("Ajouter", key=f"add_{row['conversation_pair_id']}"):
                if row['conversation_pair_id'] not in st.session_state.selected_conv_ids:
                    st.session_state.selected_conv_ids.append(row['conversation_pair_id'])
else:
    st.warning("Aucune requête ne correspond à ces critères.")
