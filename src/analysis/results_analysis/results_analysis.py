import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from config import DATA_DIR

def scores_mean_tab(df, score_cols):
    stats = df.groupby("model_id")[score_cols].agg(['mean', 'std'])
    stats.columns = [f"{col}_{stat}" for col, stat in stats.columns]
    stats = stats.reset_index()
    
    overall = df[score_cols].agg(['mean', 'std']).T
    overall.columns = [f"{stat}" for stat in overall.columns]
    overall_row = {f"{col}_{stat}": overall.loc[col, stat] for col in score_cols for stat in ['mean', 'std']}
    overall_row['model_id'] = 'TOTAL'
    stats = pd.concat([stats, pd.DataFrame([overall_row])], ignore_index=True)
    return stats

def boxplot(df, score_cols):
    num_scores = len(score_cols)
    fig, axes = plt.subplots(1, num_scores, figsize=(5 * num_scores, 6), sharey=False)
    
    for i, score in enumerate(score_cols):
        sns.boxplot(
            x="model_id",
            y=score,
            data=df,
            ax=axes[i]
        )
        axes[i].set_title(f"{score} scores by model")
        axes[i].set_xlabel("Model")
        axes[i].set_ylabel("Score Value")
        axes[i].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(DATA_DIR / f"results_analysis/box_plot_{'_'.join(score_cols)}.png")
    plt.close()
    return

if __name__ == "__main__":
    df = pd.read_csv(DATA_DIR / "results_analysis/merged_results.csv")
    score_cols = ["Method 1", "Method 2 (correction)", "Method 2 (filtering)"] 
    # score_cols = ["BERTscore (recall with ref)", "BERTscore (precision with ctx)"]
    # result = scores_mean_tab(df, score_cols)
    # results_path = DATA_DIR / "results/scores_analysis.csv"
    # result.to_csv(results_path, index=False)
    boxplot(df, score_cols)    