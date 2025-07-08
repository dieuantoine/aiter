from src.utils.utils import load_from_hf
from datasets import load_dataset, Dataset
from huggingface_hub import login
import pandas as pd

from config import HF_TOKEN, MKQA_REQ_DS, MKQA_HYP_DS, SELECTED_IDS_CSV, TEMP_HYP_CSV

tested_models = ["GPT-4o", "Le Chat", "DeepSeek"]

def expand_data(df_requests):
    expanded_rows = []
    for _, row in df_requests.iterrows():
        for model in tested_models:
            expanded_rows.append({
                "request_id": row["request_id"],
                "request": row["request"],
                "model": model,
                "response": ""
            })
    df_expanded = pd.DataFrame(expanded_rows)
    return df_expanded

def create_temp_hyp_csv():
    req_df = load_from_hf(MKQA_REQ_DS)
    selected_ids_df = pd.read_csv(SELECTED_IDS_CSV)
    req_df["request_id"] = req_df["request_id"].astype(str)
    selected_ids_df["request_id"] = selected_ids_df["request_id"].astype(str)
    df = req_df[req_df["request_id"].isin(selected_ids_df["request_id"])].copy()
    print(len(df))
    df = df[["request_id", "request"]]
    expand_data(df).to_csv(TEMP_HYP_CSV)
    return

def push_hyp_csv():
    login(HF_TOKEN)
    hyp_df = pd.read_csv(TEMP_HYP_CSV)
    hyp_ds = Dataset.from_pandas(hyp_df)
    hyp_ds.push_to_hub(MKQA_HYP_DS)
    
if __name__ == '__main__':
    create_temp_hyp_csv()