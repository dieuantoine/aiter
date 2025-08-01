import pandas as pd
from config import RESULTS_DIR, RESULTS_ANALYSIS_DIR
from src.analysis.results_analysis.results_aggregation.bertscore_computation import calculate_bertscore

def results_aggregation(df1, df2):
    base_cols = ["conv_id", "model_id", "request", "reference", "context", "hypothesis"]

    merged = pd.merge(
        df1,
        df2,
        on=["conv_id", 'model_id'],
        suffixes=('_v1', '_v2'),
        how='outer'
    )

    merged.rename(
        columns={
            "request_v1": "request",
            "hypothesis_v1": "hypothesis",
            "reference_v1": "reference",
            "context_v1": "context"
        },
        inplace=True
    )

    score_cols = [col for col in merged.columns if "score" in col]
    all_cols = base_cols + score_cols
    filtered_df = merged[all_cols]

    filtered_df['hyp_len'] = filtered_df['hypothesis'].apply(lambda x: len(x.split()))
    filtered_df['ref_len'] = filtered_df['reference'].apply(lambda x: len(x.split()))
    filtered_df['ctx_len'] = filtered_df['context'].apply(lambda x: len(x.split()))

    return filtered_df

if __name__ == "__main__":
    results_v1 = RESULTS_DIR / "results_2.csv"
    results_v2 = RESULTS_DIR / "results_1.csv"
    df1 = pd.read_csv(results_v1)
    df2 = pd.read_csv(results_v2)
    
    aggregated_results = results_aggregation(df1, df2)
    results_wbert = calculate_bertscore(aggregated_results)
    results_wbert.to_csv(RESULTS_ANALYSIS_DIR / "aggregated_results.csv", index=False)