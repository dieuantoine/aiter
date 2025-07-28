import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from config import COMPARIA_HYP_DS
from src.utils.utils import load_from_hf

def save_correlation_matrix(df):
    reaction_df = load_from_hf(COMPARIA_HYP_DS)
    df['ans_len'] = df['response'].str.split().str.len()
    merged_df = df.merge(reaction_df[['conv_id', 'liked', 'disliked', 'useful', 'creative', 'complete', 'clear_formatting', 'incorrect', 'superficial', 'instructions_not_followed']], on='conv_id', how='left')
    cols = ['score', 'ans_len', 'liked', 'disliked', 'useful', 'creative', 'complete', 'clear_formatting', 'incorrect', 'superficial', 'instructions_not_followed']
    correlation_matrix = merged_df[cols].corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Matrice de Corrélation')
    plt.savefig('correlation_matrix.png', dpi=300, bbox_inches='tight')
    