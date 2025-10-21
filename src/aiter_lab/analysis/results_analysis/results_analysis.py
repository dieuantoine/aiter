import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import median_absolute_error



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

def calculate_mae(df, hter_col, aiter_col):
    hter = df[hter_col].astype(float).to_numpy()
    aiter = df[aiter_col].astype(float).to_numpy()
    mae = np.mean(np.abs(aiter - hter))
    return mae

def trimmed_mae(df, hter_col, aiter_col, trim_ratio=0.2):
    y_true = df[hter_col].astype(float).to_numpy()
    y_pred = df[aiter_col].astype(float).to_numpy()
    errors = np.abs(y_true - y_pred)
    threshold = np.quantile(errors, 1 - trim_ratio)
    trimmed = errors[errors <= threshold]
    return trimmed.mean()

def calculate_smape(df, hter_col, aiter_col):
    hter = df[hter_col].astype(float).to_numpy()
    aiter = df[aiter_col].astype(float).to_numpy()

    denominator = (np.abs(hter) + np.abs(aiter)) / 2
    denominator[denominator == 0] = np.nan  

    smape_values = np.abs(aiter - hter) / denominator
    smape_values[np.isnan(smape_values)] = 0
    smape = np.nanmean(smape_values) * 100

    return smape

def smape_ter(df, cols):
    results = {
        'smape': {},
        'mae': {},
        'median_absolute_error': {},
        'trimmed_mae': {}
    }
    for col in cols:
        results['smape'][col] = calculate_smape(df, f'hter_{col}', col)
        results['mae'][col] = calculate_mae(df, f'hter_{col}', col)
        results['trimmed_mae'][col] = trimmed_mae(df, f'hter_{col}', col)
        results['median_absolute_error'][col] = median_absolute_error(
            df[f'hter_{col}'].astype(float).to_numpy(),
            df[col].astype(float).to_numpy()
        )
    return results