import sys, os
import streamlit as st
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../')))
from src.utils.utils import load_from_hf
from config import REQ_DS, SELECTED_IDS_CSV, VERSION

if "selected_conv_ids" not in st.session_state:
    if os.path.exists(SELECTED_IDS_CSV):
        st.session_state.selected_conv_ids = set(pd.read_csv(SELECTED_IDS_CSV)["request_id"].astype(str).tolist())
    else:
        st.session_state.selected_conv_ids = set()

@st.cache_data
def load_data():
    df = load_from_hf(REQ_DS)
    df["request_id"] = df["request_id"].astype(str)
    return df

df = load_data()

if "local_selection_updates" not in st.session_state:
    st.session_state.local_selection_updates = {}

st.title("Exploration des requêtes")

all_thematics = sorted(set(t for sublist in df["categories"] for t in sublist))
all_mots = sorted(set(m for sublist in df["question_words"] for m in sublist))

min_len, max_len = int(df["msg_length"].min()), int(df["msg_length"].max())

col1, col2 = st.columns(2)
with col1:
    msg_len_min = st.number_input("Longueur minimale du message", min_value=min_len, max_value=max_len, value=min_len, step=1)
    selected_thematiques = st.multiselect("Sélectionner une ou plusieurs thématiques", options=all_thematics)
with col2:
    msg_len_max = st.number_input("Longueur maximale du message", min_value=min_len, max_value=max_len, value=max_len, step=1)
    selected_mots = st.multiselect("Sélectionner un ou plusieurs mots interrogatifs", options=all_mots)

search_text = st.text_input("Recherche par mot-clé (optionnelle)")
show_selected_only = st.checkbox("Afficher uniquement les requêtes sélectionnées")

filtered_df = df

if selected_thematiques:
    filtered_df = filtered_df[filtered_df['categories'].apply(lambda lst: any(t in lst for t in selected_thematiques))]

if selected_mots:
    filtered_df = filtered_df[filtered_df['question_words'].apply(lambda lst: any(m in lst for m in selected_mots))]

if search_text:
    filtered_df = filtered_df[filtered_df['request'].str.contains(search_text, case=False, na=False)]

if show_selected_only:
    filtered_df = filtered_df[filtered_df['request_id'].isin(st.session_state.selected_conv_ids)]

if msg_len_min > msg_len_max:
    st.warning("La longueur minimale est supérieure à la longueur maximale.")
else:
    filtered_df = filtered_df[
        (filtered_df["msg_length"] >= msg_len_min) &
        (filtered_df["msg_length"] <= msg_len_max)
    ]

items_per_page = 20
total_results = len(filtered_df)
total_pages = (total_results - 1) // items_per_page + 1

if total_results > 0:
    page = st.number_input("Page", min_value=1, max_value=total_pages, value=1, step=1)
    start_idx = (page - 1) * items_per_page
    end_idx = start_idx + items_per_page

    st.write(f"Résultats {start_idx + 1} à {min(end_idx, total_results)} sur {total_results}")
    for idx, row in filtered_df.iloc[start_idx:end_idx].iterrows():
        cols = st.columns([10, 2])
        with cols[0]:
            full_words = row['request'].split()
            if len(full_words) > 100:
                short_msg = ' '.join(full_words[:100]) + "..."
            else:
                short_msg = row['request']
            st.markdown(short_msg)

            with st.expander("➕ Plus d'informations"):
                st.markdown(f"id: {row['request_id']}")
                st.markdown(f"**Thématiques :** {', '.join(row['categories'])}")
                st.markdown(f"**Mot(s) interrogatif(s) :** {', '.join(row['question_words'])}")
                st.markdown(f"**Requête ({row['msg_length']}) :** {row['request']}")
                if VERSION['DATASET_VERSION']=="mkqa":
                    st.markdown(f"**Annotations :** {row['reference_annotation']}")
    
        with cols[1]:
            conv_id = row['request_id']

            base_state = conv_id in st.session_state.selected_conv_ids
            updated_state = st.session_state.local_selection_updates.get(conv_id, base_state)

            btn_label = "➖ Retirer" if updated_state else "➕ Ajouter"

            if st.button(btn_label, key=f"toggle_{conv_id}_{updated_state}"):
                st.session_state.local_selection_updates[conv_id] = not updated_state
                st.rerun()
else:
    st.warning("Aucune requête ne correspond à ces critères.")

if st.button("💾 Enregistrer les modifications"):
    if os.path.exists(SELECTED_IDS_CSV):
        all_ids = set(pd.read_csv(SELECTED_IDS_CSV)["request_id"].tolist())
    else:
        all_ids = set()

    added = 0
    removed = 0

    for conv_id, new_state in st.session_state.local_selection_updates.items():
        if new_state and conv_id not in all_ids:
            all_ids.add(conv_id)
            added += 1
        elif not new_state and conv_id in all_ids:
            all_ids.discard(conv_id)
            removed += 1

    pd.DataFrame({"request_id": list(all_ids)}).to_csv(SELECTED_IDS_CSV, index=False)
    
    st.session_state.selected_conv_ids = all_ids
    st.session_state.local_selection_updates = {}
    st.success(f"Sélection mise à jour : {added} entrée(s) ajoutée(s), {removed} retirée(s).")