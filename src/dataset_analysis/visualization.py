from huggingface_hub import login
from datasets import load_dataset
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

def plot_msg_length(df, cap=True):
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
    plt.show()
    return

def create_wc(word_counts):
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(word_counts)
    plt.figure(figsize=(12,6))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.show()
    return

def create_pie(counts):
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
    plt.show()
    return
