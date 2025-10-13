from ..utils import load_from_hf, push_to_hf
import pandas as pd

def init_temp_hyp_csv(mkqa_hyp_ds_path, selected_ids_csv_path, temp_hyp_csv):
    hyp_df = load_from_hf(mkqa_hyp_ds_path)
    selected_ids_df = pd.read_csv(selected_ids_csv_path)
    
    hyp_df["request_id"] = hyp_df["request_id"].astype(str)
    selected_ids_df["request_id"] = selected_ids_df["request_id"].astype(str)
    
    filtered_df = hyp_df[hyp_df["request_id"].isin(selected_ids_df["request_id"])].copy()
    filtered_df = filtered_df[
        filtered_df["hypothesis"].isna() | (filtered_df["hypothesis"].astype(str).str.strip() == '')
    ]

    filtered_df[['conv_id', 'model_id', 'request_id', 'request', 'hypothesis']].to_csv(temp_hyp_csv, index=False)
    return
    
    
def push_temp_hyp_csv(mkqa_hyp_ds_path, temp_hyp_csv):
    temp_hyp_df = pd.read_csv(temp_hyp_csv)
    existing_hyp_df = load_from_hf(mkqa_hyp_ds_path)
    
    merged_df = pd.merge(
        existing_hyp_df,
        temp_hyp_df,
        on=['conv_id', 'model_id', 'request_id', 'request'],
        how='left',
        suffixes=('', '_new')
    )
    
    merged_df['hypothesis'] = merged_df.apply(
        lambda row: row['hypothesis_new'] if pd.notna(row['hypothesis_new']) and str(row['hypothesis_new']).strip() != '' else row['hypothesis'],
        axis=1
    )
    
    hyp_df = merged_df.drop(columns=['hypothesis_new'])

    push_to_hf(hyp_df, mkqa_hyp_ds_path)
    return
    
    