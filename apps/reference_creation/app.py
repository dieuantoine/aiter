import streamlit as st
import pandas as pd
import os, sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from src.aiter_lab.config import REQ_DS, HYP_DS, SELECTED_IDS_CSV, VERSION, REFERENCES_CSV
from src.aiter_lab.utils import load_from_hf

@st.cache_data
def load_data():
    ids_df = pd.read_csv(SELECTED_IDS_CSV)
    ids_df["request_id"] = ids_df["request_id"].astype(str)
    df = load_from_hf(HYP_DS)
    df["request_id"] = df["request_id"].astype(str)
    if VERSION['DATASET'] == "mkqa":
        resp_df = load_from_hf(REQ_DS)
    else:
        resp_df = None
    request_ids = set(ids_df['request_id'].tolist())
    return df, request_ids, resp_df

df, request_ids, resp_df = load_data()

if "current_index" not in st.session_state:
    st.session_state.current_index = 0

if "annotations_df" not in st.session_state:
    try:
        st.session_state.annotations_df = pd.read_csv(REFERENCES_CSV)
    except FileNotFoundError:
        st.session_state.annotations_df = pd.DataFrame(columns=["request_id", "request", "reference", "context", "reference_created"])

if "remaining_ids" not in st.session_state:
    annotated_ids = set(st.session_state.annotations_df[st.session_state.annotations_df['reference_created'] == 1]['request_id'])
    st.session_state.remaining_ids = list(request_ids - annotated_ids)

if not st.session_state.remaining_ids:
    st.success("All data have been annotated.")
    st.stop()

if st.session_state.current_index >= len(st.session_state.remaining_ids):
    st.session_state.current_index = 0

current_request_id = st.session_state.remaining_ids[st.session_state.current_index]
request_group = df[df['request_id'] == current_request_id]
request_text = request_group['request'].iloc[0]
responses = request_group['hypothesis'].tolist()

st.title("Annotation Interface")

st.subheader("Request :")
st.markdown(f"> {request_text}")

if VERSION['DATASET'] == "mkqa":
    st.subheader("MKQA Reference Elements:")
    ref_text_df = resp_df[resp_df['request_id'] == str(current_request_id)]
    ref_text = ref_text_df['reference_annotation'].iloc[0] if not ref_text_df.empty else None
    st.markdown(ref_text)

st.subheader("Model Hypotheses:")
for idx, resp in enumerate(responses, 1):
    st.markdown(f"**Réponse {idx} :** {resp}")


ref_key = f"ref_{current_request_id}"
ctx_key = f"ctx_{current_request_id}"

existing_row = st.session_state.annotations_df[st.session_state.annotations_df['request_id'] == current_request_id]
if not existing_row.empty:
    existing_ref = existing_row['reference'].values[0]
    existing_ctx = existing_row['context'].values[0]
else:
    existing_ref = ""
    existing_ctx = ""

st.subheader("Enter your reference:")
reference = st.text_area("Your text here", value=st.session_state.get(ref_key, existing_ref), height=200)
st.session_state[ref_key] = reference

st.subheader("Enter the associated context:")
context = st.text_area("Your text here", value=st.session_state.get(ctx_key, existing_ctx), height=150)
st.session_state[ctx_key] = context

col1, col2, col3 = st.columns([1, 1, 2])
with col1:
    if st.button("⬅️ Previous", disabled=st.session_state.current_index == 0) and st.session_state.current_index > 0:
        st.session_state.current_index -= 1
        st.rerun()
with col2:
    if st.button("Next ➡️", disabled=st.session_state.current_index == len(st.session_state.remaining_ids)) and st.session_state.current_index < len(st.session_state.remaining_ids) - 1:
        st.session_state.current_index += 1
        st.rerun()
with col3:
    st.write(f"Current reference: {st.session_state.current_index + 1}/{len(st.session_state.remaining_ids)}")

if st.button("Submit"):
    if reference.strip() == "" or context.strip() == "":
        st.warning("Please fill in both the reference and the context.")
    else:
        new_entry = {
            "request_id": current_request_id,
            "request": request_text,
            "reference": reference.strip(),
            "context": context.strip(),
            "reference_created": 1
        }
        
        existing = st.session_state.annotations_df['request_id'] == current_request_id
        if existing.any():
            st.session_state.annotations_df = st.session_state.annotations_df[~existing]

        st.session_state.annotations_df = pd.concat(
            [st.session_state.annotations_df, pd.DataFrame([new_entry])],
            ignore_index=True
        )
        st.session_state.annotations_df.to_csv(REFERENCES_CSV, index=False)

        st.session_state.remaining_ids.pop(st.session_state.current_index)
        
        if st.session_state.current_index >= len(st.session_state.remaining_ids):
            st.session_state.current_index = 0

        st.success("Reference saved.")
        st.rerun()
