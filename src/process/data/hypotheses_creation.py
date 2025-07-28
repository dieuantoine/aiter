from src.utils.utils import load_from_hf
from datasets import Dataset
from huggingface_hub import login
import pandas as pd

from config import HF_TOKEN, TESTED_MODELS

def expand_data(df_requests):
    expanded_rows = []
    conv_id = 0
    for _, row in df_requests.iterrows():
        for model in TESTED_MODELS:
            expanded_rows.append({
                "conv_id": conv_id,
                "request_id": str(row["request_id"]),
                "request": row["request"],
                "model_id": model,
                "response": ""
            })
            conv_id += 1
    df_expanded = pd.DataFrame(expanded_rows)
    return df_expanded

def create_temp_hyp_csv(mkqa_req_ds_path, selected_ids_csv_path, temp_hyp_csv):
    req_df = load_from_hf(mkqa_req_ds_path)
    selected_ids_df = pd.read_csv(selected_ids_csv_path)
    req_df["request_id"] = req_df["request_id"].astype(str)
    selected_ids_df["request_id"] = selected_ids_df["request_id"].astype(str)
    df = req_df[req_df["request_id"].isin(selected_ids_df["request_id"])].copy()
    df = df[["request_id", "request"]]
    expand_data(df)[['conv_id', 'model_id', 'request_id', 'request', 'response']].to_csv(temp_hyp_csv, index=False)
    return

def push_hyp_csv(mkqa_hyp_ds_path, temp_hyp_ds_path):
    login(HF_TOKEN)
    hyp_df = pd.read_csv(temp_hyp_ds_path)
    hyp_ds = Dataset.from_pandas(hyp_df)
    hyp_ds.push_to_hub(mkqa_hyp_ds_path)
    return