import pandas as pd
from config import DATA_DIR


results_1 = pd.read_csv(DATA_DIR / "results/results_hyp_3.csv")
results_2 = pd.read_csv( DATA_DIR / "results/results_1_w_bertscore.csv")

merged = pd.merge(
    results_1,
    results_2,
    on=["conv_id", "model_id", "request"],
    suffixes=('_1', '_2'),
    how='outer'
)

score_cols = [col for col in merged.columns if "score" in col]
columns_to_keep = [
    "conv_id",
    "model_id",
    "request",
    "hypothesis",
    "reference",
    "context"
] + score_cols

merged.rename(columns={
    "reference_2": "reference"
}, inplace=True)

filtered_df = merged[columns_to_keep]

filtered_df['hyp_len'] = filtered_df['hypothesis'].apply(lambda x: len(x.split()))
filtered_df['ref_len'] = filtered_df['reference'].apply(lambda x: len(x.split()))
filtered_df['ctx_len'] = filtered_df['context'].apply(lambda x: len(x.split()))

filtered_df.to_csv(DATA_DIR / "results/merged_results.csv", index=False)
