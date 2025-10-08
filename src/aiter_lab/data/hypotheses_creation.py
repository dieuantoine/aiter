from ..utils import load_from_hf
from datasets import Dataset
from huggingface_hub import login
import pandas as pd
import os

from config import HF_TOKEN, TESTED_MODELS

def expand_data(df_requests, existing_df):
    existing_ids = set(zip(existing_df['request_id'].astype(str), existing_df['model_id']))
    new_rows = []
    conv_id = len(existing_ids)
    for _, row in df_requests.iterrows():
        req_id = str(row["request_id"])
        for model in TESTED_MODELS:
            if (req_id, model) not in existing_ids:
                new_rows.append({
                    "conv_id": conv_id,
                    "model_id": model,
                    "request_id": req_id,
                    "request": row["request"],
                    "hypothesis": ""
            })
                conv_id += 1
    new_df = pd.DataFrame(new_rows)
    df_expanded = pd.concat([existing_df, new_df], ignore_index=True)
    return df_expanded

def create_temp_hyp_csv(mkqa_req_ds_path, selected_ids_csv_path, temp_hyp_csv):
    req_df = load_from_hf(mkqa_req_ds_path)
    selected_ids_df = pd.read_csv(selected_ids_csv_path)
    
    req_df["request_id"] = req_df["request_id"].astype(str)
    selected_ids_df["request_id"] = selected_ids_df["request_id"].astype(str)
    
    df = req_df[req_df["request_id"].isin(selected_ids_df["request_id"])].copy()
    df = df[["request_id", "request"]]

    if os.path.exists(temp_hyp_csv):
        existing_df = pd.read_csv(temp_hyp_csv)
    else:
        existing_df = pd.DataFrame(columns=['conv_id', 'model_id', 'request_id', 'request', 'hypothesis'])
    df = expand_data(df, existing_df)
    df[['conv_id', 'model_id', 'request_id', 'request', 'hypothesis']].to_csv(temp_hyp_csv, index=False)
    return

def push_hyp_csv(mkqa_hyp_ds_path, temp_hyp_ds_path):
    login(HF_TOKEN)
    hyp_df = pd.read_csv(temp_hyp_ds_path)
    hyp_ds = Dataset.from_pandas(hyp_df)
    hyp_ds.push_to_hub(mkqa_hyp_ds_path)
    return