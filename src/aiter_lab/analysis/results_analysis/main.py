from .results_analysis import scores_mean_tab, boxplot, correlation_matrix, smape_ter
from ...config import DATA_DIR, RESULTS_ANALYSIS_DIR
import pandas as pd

if __name__ == "__main__":
    df = pd.read_csv(RESULTS_ANALYSIS_DIR / "method3_fullresults.csv")
    print(smape_ter(df, ["score", "cor_score", "ot_score"]))
    df.rename(columns={
        "score": "Method 3 (global)",
        "cor_score": "Method 3 (correction)",
        "ot_score": "Method 3 (filtering)",
        "hter_score": "HTER Score",
        "hter_ot_score": "HTER (filtering)",
        "hter_cor_score": "HTER (correction)",
        "bertscore_recall_ref": "BERTRecall (ref)",
        "bertscore_precision_ctx": "BERTPrecision (ctx)",
        "hyp_len": "Hypothesis Length"
    }, inplace=True)
    
    models = df["model_id"].unique()

    score_cols = ["Method 3 (correction)", "Method 3 (filtering)", "Method 3 (global)", "HTER Score", "HTER (correction)", "HTER (filtering)"]
    bertscore_cols = ["BERTRecall (ref)", "BERTPrecision (ctx)"]
    cols = score_cols + bertscore_cols + ["Hypothesis Length"]
    
    # results_path = RESULTS_ANALYSIS_DIR / "scores_analysis.csv"
    # scores_mean_tab(df, score_cols+bertscore_cols, results_path)

    # bertscore_boxplot_path = RESULTS_ANALYSIS_DIR / "bertscore_boxplot.png"
    # boxplot(df, bertscore_cols, bertscore_boxplot_path)

    # score_boxplot_path = RESULTS_ANALYSIS_DIR / "aiter_boxplot.png"
    # boxplot(df, score_cols, score_boxplot_path)
    
    # for model in models:
    #     model_df = df[df["model_id"] == model]
    #     results_path = RESULTS_ANALYSIS_DIR / f"correlation_matrix_{model}_v3.png"
    #     correlation_matrix(model_df, cols, results_path)
    
    # corr_matrix_path = RESULTS_ANALYSIS_DIR / "correlation_matrix.png"
    # correlation_matrix(df, cols, corr_matrix_path)

