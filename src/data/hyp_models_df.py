from src.utils.utils import load_from_hf
import pandas as pd

from config import CONV_DS, DATA_DIR, MKQA_REQ_DS, SELECTED_IDS_CSV

models = ["GPT-4o", "Le Chat", "DeepSeek"]

def expand_data(df_requests):
    expanded_rows = []
    for _, row in df_requests.iterrows():
        for model in models:
            expanded_rows.append({
                "request_id": row["request_id"],
                "request": row["request"],
                "model": model,
                "response": ""
            })
    df_expanded = pd.DataFrame(expanded_rows)
    return df_expanded

def main():
    req_df = load_from_hf(MKQA_REQ_DS)
    selected_ids_df = pd.read_csv(DATA_DIR / SELECTED_IDS_CSV)
    req_df["request_id"] = req_df["request_id"].astype(str)
    selected_ids_df["request_id"] = selected_ids_df["request_id"].astype(str)
    df = req_df[req_df["request_id"].isin(selected_ids_df["request_id"])].copy()
    print(len(df))
    df = df[["request_id", "request"]]
    expand_data(df).to_csv(DATA_DIR / 'hyp_1.csv')
    
if __name__ == '__main__':
    main()