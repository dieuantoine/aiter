import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

def plot_msg_length(df, output_path, cap=True):
    if cap:
        word_95 = df['msg_length'].quantile(0.95)
        char_95 = df['char_length'].quantile(0.95)
        df['msg_length'] = df['msg_length'].apply(lambda x: min(x, word_95))
        df['char_length'] = df['char_length'].apply(lambda x: min(x, char_95))
    def cap_name(title, cap):
        if cap:
            return title+" (Capped)"
        else:
            return title
    fig, axes = plt.subplots(2, 1, figsize=(10, 10), sharex=False)
    axes[0].hist(df['msg_length'], bins=50, color='skyblue', edgecolor='black')
    axes[0].set_title(cap_name("Word Count Histogram", cap))
    axes[0].set_xlabel("Word Count")
    axes[0].set_ylabel("Frequency")
    axes[0].grid(True)
    axes[1].hist(df['char_length'], bins=50, color='orange', edgecolor='black')
    axes[1].set_title(cap_name("Character Count Histogram", cap))
    axes[1].set_xlabel("Character Count")
    axes[1].set_ylabel("Frequency")
    axes[1].grid(True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close(fig)
    return

def compare_msg_length(df1, df2, labels, output_path, cap=100):
    data1 = df1['msg_length'].apply(lambda x: min(x, cap))
    data2 = df2['msg_length'].apply(lambda x: min(x, cap))
    plt.figure(figsize=(10, 6))
    sns.kdeplot(data1, label=labels[0], fill=True, alpha=0.5, color='skyblue')
    sns.kdeplot(data2, label=labels[1], fill=True, alpha=0.5, color='salmon')
    plt.title("Smoothed Comparison of Message Lengths (KDE) - Capped")
    plt.xlabel("Message Length (Words)")
    plt.ylabel("Density")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    return

def compare_hyp_length(df, cols, labels, output_path):
    plt.figure(figsize=(10, 6))
    for col, label in zip(cols, labels):
        sns.kdeplot(df[col], label=label, fill=True, alpha=0.5)
    plt.title("Smoothed Comparison of Hypothesis Lengths (KDE)")
    plt.xlabel("Hypothesis Length (Words)")
    plt.ylabel("Density")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    return

def create_wc(counts, output_path):
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(counts)
    plt.figure(figsize=(12,6))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.savefig(output_path, dpi=300)
    plt.close()
    return

def create_pie(counts, output_path):
    sorted_items = sorted(counts.items(), key=lambda x: x[1])
    labels = [item[0] for item in sorted_items]
    sizes = [item[1] for item in sorted_items]
    total = sum(sizes)
    def make_autopct(sizes):
        def autopct(pct):
            count = int(round(pct * total / 100.0))
            return f"{count} ({pct:.1f}%)"
        return autopct
    plt.figure(figsize=(8, 8))
    plt.pie(sizes, labels=labels, autopct=make_autopct(sizes), startangle=140)
    plt.title("Répartition")
    plt.axis('equal')
    plt.savefig(output_path, dpi=300)
    plt.close()
    return

def weekly_occurence_counts(conv_df):
    conv_df['date'] = pd.to_datetime(conv_df['timestamp']).dt.date
    conv_df['semaine'] = conv_df['timestamp'].dt.to_period('W')
    df_exploded = conv_df.explode('categories')
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
    return