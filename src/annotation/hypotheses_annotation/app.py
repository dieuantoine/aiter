import streamlit as st
import pandas as pd
import os, sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))
from config import TEMP_HYP_CSV

st.title("📝 Annotation des réponses par modèle")

if "df" not in st.session_state:
    if os.path.exists(TEMP_HYP_CSV):
        st.session_state.df = pd.read_csv(TEMP_HYP_CSV)
    else:
        st.error("Le fichier annoté n'existe pas.")
        st.stop()

df = st.session_state.df

df_todo = df[df["response"].isna()]

if df_todo.empty:
    st.success("Toutes les requêtes ont été annotées pour tous les modèles.")
    st.stop()

selected_index = st.selectbox(
    "Sélectionnez une requête/modèle à annoter :",
    df_todo.index,
    format_func=lambda i: f"ID {df_todo.at[i, 'request_id']} | {df_todo.at[i, 'model']} | {df_todo.at[i, 'request'][:60]}...",
    key="selected_index"
)

selected_row = df_todo.loc[selected_index]

st.markdown(f"### Requête ID : `{selected_row['request_id']}`")
st.markdown(f"**Modèle :** `{selected_row['model']}`")
st.info(selected_row["request"])

response_key = f"response_text_{selected_index}"
response_text = st.text_area("Entrez la réponse du modèle :", height=200, key=response_key)

if st.button("Enregistrer la réponse"):
    if not response_text.strip():
        st.warning("La réponse ne peut pas être vide.")
    else:
        st.session_state.df.at[selected_index, "response"] = response_text.strip()
        st.session_state.df.to_csv(TEMP_HYP_CSV, index=False)
        st.success("Réponse enregistrée.")
        del st.session_state["selected_index"]
        del st.session_state[response_key]

st.markdown(f"**Requêtes restantes :** {len(df_todo)}")
