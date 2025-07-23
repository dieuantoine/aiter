import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from config import DATA_DIR

def correlation_matrix(df, cols):
    corr_matrix = df[cols].corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm")
    plt.title("Correlation Matrix")
    plt.tight_layout()
    plt.savefig(DATA_DIR / "results_analysis/correlation_matrix.png")
    plt.close()
    return corr_matrix
    
if __name__ == "__main__":
    df = pd.read_csv(DATA_DIR / "results_analysis/merged_results.csv")
    cols = [
        "bertrecall_ref",
        "bertprecision_ctx",
        "score_1",
        "score_2_cor",
        "score_2_fil",
        "hyp_len"
    ]
    # Generate and save the correlation matrix
    correlation_matrix(df, cols)