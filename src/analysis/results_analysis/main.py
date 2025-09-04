from src.analysis.results_analysis.results_analysis import scores_mean_tab, boxplot, correlation_matrix
from config import DATA_DIR, RESULTS_ANALYSIS_DIR
import pandas as pd

if __name__ == "__main__":
    df = pd.read_csv(RESULTS_ANALYSIS_DIR / "aggregated_results_v13.csv")
    df.rename(columns={
        "score_v1": "Method 1",
        "cor_score": "Method 3 (correction)",
        "ot_score": "Method 3 (filtering)",
        "score_v3": "Method 3 (global)",
        "bertscore_recall_ref": "BERTRecall (ref)",
        "bertscore_precision_ctx": "BERTPrecision (ctx)",
        "hyp_len": "Hypothesis Length"
    }, inplace=True)
    
    models = df["model_id"].unique()

    score_cols = ["Method 1", "Method 3 (correction)", "Method 3 (filtering)", "Method 3 (global)"]
    bertscore_cols = ["BERTRecall (ref)", "BERTPrecision (ctx)"]
    cols = score_cols + bertscore_cols + ["Hypothesis Length"]
    
    # results_path = RESULTS_ANALYSIS_DIR / "scores_analysis.csv"
    # scores_mean_tab(df, score_cols+bertscore_cols, results_path)

    # bertscore_boxplot_path = RESULTS_ANALYSIS_DIR / "bertscore_boxplot.png"
    # boxplot(df, bertscore_cols, bertscore_boxplot_path)

    # score_boxplot_path = RESULTS_ANALYSIS_DIR / "aiter_boxplot_v3.png"
    # boxplot(df, score_cols, score_boxplot_path)
    
    # for model in models:
    #     model_df = df[df["model_id"] == model]
    #     results_path = RESULTS_ANALYSIS_DIR / f"correlation_matrix_{model}_v3.png"
    #     correlation_matrix(model_df, cols, results_path)
    
    corr_matrix_path = RESULTS_ANALYSIS_DIR / "correlation_matrix_v3.png"
    correlation_matrix(df, cols, corr_matrix_path)