import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from config import DATA_DIR

def scores_mean_tab(df, score_cols, output_path):
    stats = df.groupby("model_id")[score_cols].agg(['mean', 'std'])
    stats.columns = [f"{col}_{stat}" for col, stat in stats.columns]
    stats = stats.reset_index()
    
    overall = df[score_cols].agg(['mean', 'std']).T
    overall.columns = [f"{stat}" for stat in overall.columns]
    overall_row = {f"{col}_{stat}": overall.loc[col, stat] for col in score_cols for stat in ['mean', 'std']}
    overall_row['model_id'] = 'TOTAL'
    stats = pd.concat([stats, pd.DataFrame([overall_row])], ignore_index=True)
    stats.to_csv(output_path, index=False)
    return

def correlation_matrix(df, cols, output_path):
    corr_matrix = df[cols].corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm")
    plt.title("Correlation Matrix")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    return corr_matrix

def boxplot(df, score_cols, output_path):
    num_scores = len(score_cols)
    fig, axes = plt.subplots(1, num_scores, figsize=(5 * num_scores, 6), sharey=False)
    
    for i, score in enumerate(score_cols):
        sns.boxplot(
            x="model_id",
            y=score,
            data=df,
            ax=axes[i],
            showfliers=False,
            palette="Set2"
        )
        axes[i].set_title(f"{score} scores by model")
        axes[i].set_xlabel("Model")
        axes[i].set_ylabel("Score Value")
        axes[i].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    return