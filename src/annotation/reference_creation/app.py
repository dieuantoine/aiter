import streamlit as st
import pandas as pd
import os, sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))

from src.utils.utils import load_from_hf
from config import SELECTED_IDS_CSV, REFERENCES_CSV, HYP_DS

ids_df = pd.read_csv(SELECTED_IDS_CSV)
request_ids = set(ids_df['request_id'].tolist())

df = load_from_hf(HYP_DS)

try:
    annotations_df = pd.read_csv(REFERENCES_CSV)
except FileNotFoundError:
    annotations_df = pd.DataFrame(columns=["request_id", "request", "reference", "reference_created"])
    
annotated_ids = set(annotations_df[annotations_df['reference_created'] == 1]['request_id'])
remaining_ids = list(request_ids - annotated_ids)

if not remaining_ids:
    st.success("Toutes les données ont été annotées !")
    st.stop()
    
current_request_id = remaining_ids[0]
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
        annotations_df = pd.concat([annotations_df, pd.DataFrame([new_entry])], ignore_index=True)

        annotations_df.to_csv(REFERENCES_CSV, index=False)

        st.success("Référence enregistrée ! Rechargez la page pour continuer.")