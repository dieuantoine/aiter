import streamlit as st
import pandas as pd
import os, sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))
from src.utils.utils import load_from_hf
from config import SELECTED_IDS_CSV, REFERENCES_CSV, HYP_DS

@st.cache_data
def load_data():
    ids_df = pd.read_csv(SELECTED_IDS_CSV)
    df = load_from_hf(HYP_DS)
    request_ids = set(ids_df['request_id'].tolist())
    return df, request_ids

df, request_ids = load_data()

if "current_index" not in st.session_state:
    st.session_state.current_index = 0

if "annotations_df" not in st.session_state:
    try:
        st.session_state.annotations_df = pd.read_csv(REFERENCES_CSV)
    except FileNotFoundError:
        st.session_state.annotations_df = pd.DataFrame(columns=["request_id", "request", "reference", "reference_created"])

if "remaining_ids" not in st.session_state:
    annotated_ids = set(st.session_state.annotations_df[st.session_state.annotations_df['reference_created'] == 1]['request_id'])
    st.session_state.remaining_ids = list(request_ids - annotated_ids)

if not st.session_state.remaining_ids:
    st.success("Toutes les données ont été annotées !")
    st.stop()

if st.session_state.current_index >= len(st.session_state.remaining_ids):
    st.session_state.current_index = 0
current_request_id = st.session_state.remaining_ids[st.session_state.current_index]
request_group = df[df['request_id'] == current_request_id]
request_text = request_group['request'].iloc[0]
responses = request_group['response'].tolist()

st.title("Interface d'annotation")

st.subheader("Requête :")
st.markdown(f"> {request_text}")

st.subheader("Réponses :")
for idx, resp in enumerate(responses, 1):
    st.markdown(f"**Réponse {idx} :** {resp}")

st.subheader("Entrez votre référence :")
reference = st.text_area("Votre texte ici", height=200)

col1, col2, col3 = st.columns([1, 1, 2])
with col1:
    if st.button("⬅️ Précédent") and st.session_state.current_index > 0:
        st.session_state.current_index -= 1
        st.experimental_rerun()
with col2:
    if st.button("Suivant ➡️") and st.session_state.current_index < len(st.session_state.remaining_ids) - 1:
        st.session_state.current_index += 1
        st.experimental_rerun()

if st.button("Soumettre"):
    if reference.strip() == "":
        st.warning("Veuillez entrer un texte.")
    else:
        new_entry = {
            "request_id": current_request_id,
            "request": request_text,
            "reference": reference.strip(),
            "reference_created": 1
        }

        st.session_state.annotations_df = pd.concat(
            [st.session_state.annotations_df, pd.DataFrame([new_entry])],
            ignore_index=True
        )
        st.session_state.annotations_df.to_csv(REFERENCES_CSV, index=False)

        st.session_state.remaining_ids.pop(0)
        
        if st.session_state.current_index >= len(st.session_state.remaining_ids):
            st.session_state.current_index = 0

        st.success("Référence enregistrée !")
        st.experimental_rerun()
