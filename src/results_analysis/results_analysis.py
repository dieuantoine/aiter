import pandas as pd

from config import DATA_DIR

def scores_analysis(df, score_cols):
    stats = df.groupby("model_id")[score_cols].agg(['mean', 'std'])
    stats.columns = [f"{col}_{stat}" for col, stat in stats.columns]
    stats = stats.reset_index()
    
    overall = df[score_cols].agg(['mean', 'std']).T
    overall.columns = [f"{stat}" for stat in overall.columns]
    overall_row = {f"{col}_{stat}": overall.loc[col, stat] for col in score_cols for stat in ['mean', 'std']}
    overall_row['model_id'] = 'TOTAL'
    stats = pd.concat([stats, pd.DataFrame([overall_row])], ignore_index=True)
    return stats

if __name__ == "__main__":
    df = pd.read_csv(DATA_DIR / "results/merged_results.csv")
    score_cols = ["score_1", "score_2", "ot_score", "bertscore_recall_ref", "bertscore_precision_ctx"]
    result = scores_analysis(df, score_cols)
    results_path = DATA_DIR / "results/scores_analysis.csv"
    result.to_csv(results_path, index=False)