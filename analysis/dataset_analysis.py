from huggingface_hub import login
from datasets import load_dataset
import matplotlib.pyplot as plt
import pandas as pd
# from analysis.counters import words_counter, list_features_counter
# from analysis.visualization import plot_msg_length, create_pie
from config import HF_TOKEN, CONV_DS

if __name__ == '__main__':
    login(HF_TOKEN)
    df = load_dataset(CONV_DS)['train'].to_pandas()
    #counts = list_features_counter(df, 'categories')
    #create_pie(counts)
    df['date'] = pd.to_datetime(df['timestamp']).dt.date
    df['semaine'] = df['timestamp'].dt.to_period('W')
    df_exploded = df.explode('categories')
    weekly_category_counts = (
        df_exploded
        .groupby(['semaine', 'categories'])
        .size()
        .reset_index(name='count')
    )
    pivot_df = weekly_category_counts.pivot(index='semaine', columns='categories', values='count').fillna(0)

    plt.figure(figsize=(14, 6))
    pivot_df.plot(kind='line', figsize=(14, 6), marker='o')

    plt.title("Nombre d'entrées par semaine pour chaque catégorie")
    plt.xlabel("Semaine")
    plt.ylabel("Nombre d'entrées")
    plt.legend(title="Catégories", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    # weekly_counts = df['semaine'].value_counts().sort_index()
    # plt.figure(figsize=(12,6))
    # weekly_counts.plot(kind='bar')
    # plt.title("Nombre de lignes par semaine")
    # plt.xlabel("Semaine")
    # plt.ylabel("Nombre de lignes")
    # plt.xticks(rotation=45)
    # plt.tight_layout()
    # plt.show()