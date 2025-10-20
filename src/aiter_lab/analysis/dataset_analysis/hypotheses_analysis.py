import pandas as pd
from ...utils import load_from_hf
from ...config import HYP_DS, REFERENCES_CSV
import matplotlib.pyplot as plt
import seaborn as sns

def hyp_len_stats_tab(input_df):
    input_df = input_df[input_df['hypothesis'].str.len() > 0].copy()
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
    hyp_df = hyp_len_stats_tab(load_from_hf(HYP_DS))
    ref_df = ref_len_stats_tab(pd.read_csv(REFERENCES_CSV))
    cols = ['source', 'mean', 'std', 'min', '25%', '50%', '75%', 'max']
    merged_df = pd.concat([hyp_df[cols], ref_df[cols]])
    merged_df.to_csv(output_csv, index=False)
    return

def len_correlations(input_df, output_path):
    input_df['ref_length'] = input_df['reference'].apply(lambda x: len(x.split()))
    input_df['ctx_length'] = input_df['context'].apply(lambda x: len(x.split()))
    input_df['hyp_length'] = input_df['hypothesis'].apply(lambda x: len(x.split()))
    def correlation_reference_hypothese(group):
        corr_ref = group['ref_length'].corr(group['hyp_length'])
        corr_ctx = group['ctx_length'].corr(group['hyp_length'])
        corr = pd.Series({
            'Reference': corr_ref,
            'Context': corr_ctx
        })
        return corr
    correlation_matrix = input_df.groupby('model_id', group_keys=False).apply(correlation_reference_hypothese).T
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', cbar=False)
    plt.title('Correlation matrix')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    return