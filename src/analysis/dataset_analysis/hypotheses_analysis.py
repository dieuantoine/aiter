import pandas as pd
from config import MKQA_HYP_DS, REFERENCES_CSV
from src.utils.utils import load_from_hf

def hyp_len_stats_tab(input_df):
    #len_mean_tab(input_df).to_csv(output_csv, index=False)
    input_df['hyp_length'] = input_df['hypothesis'].apply(lambda x: len(x.split()))
    hyp_df = input_df.groupby("model_id")["hyp_length"].describe().sort_values("mean")
    hyp_df = hyp_df.reset_index()
    hyp_df['source'] = hyp_df['model_id']
    return hyp_df

def ref_len_stats_tab(input_df):
    input_df['ref_length'] = input_df['reference'].apply(lambda x: len(x.split()))
    input_df['ctx_length'] = input_df['context'].apply(lambda x: len(x.split()))
    ref_row = input_df["ref_length"].describe().to_frame().T
    ctx_row = input_df["ctx_length"].describe().to_frame().T
    ref_row['source'] = ["Reference"]
    ctx_row['source'] = ["Context"]
    stats_df = pd.concat([ref_row, ctx_row])
    return stats_df

def merge_stats_tab(output_csv):
    hyp_df = hyp_len_stats_tab(load_from_hf(MKQA_HYP_DS))
    ref_df = ref_len_stats_tab(pd.read_csv(REFERENCES_CSV))
    cols = ['source', 'mean', 'std', 'min', '25%', '50%', '75%', 'max']
    merged_df = pd.concat([hyp_df[cols], ref_df[cols]])
    merged_df.to_csv(output_csv, index=False)
    return