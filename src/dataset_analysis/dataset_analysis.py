from huggingface_hub import login
from datasets import load_dataset
import pandas as pd

from src.dataset_analysis.counters import words_counter, categories_counter

import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

from config import HF_TOKEN, REQ_DS, question_words

def calc_msg_length(df, cap=True):
    df['msg_length'] = df['opening_msg'].apply(lambda x: len(str(x).split()))
    df['char_length'] = df['opening_msg'].apply(lambda x: len(str(x)))
    if cap:
        word_95 = df['msg_length'].quantile(0.95)
        char_95 = df['char_length'].quantile(0.95)
        df['msg_length'] = df['msg_length'].apply(lambda x: min(x, word_95))
        df['char_length'] = df['char_length'].apply(lambda x: min(x, char_95))
    return df

def plot_msg_length(df, cap=True):
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

def create_wc(word_counts, k=20):
    common_words = word_counts.most_common(k)
    print(common_words)
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(word_counts)
    plt.figure(figsize=(12,6))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.show()
    return

def calc_qw(df):
    df['question_word'] = df['opening_msg'].apply(
    lambda x: next((w for w in question_words if w in x.split()), None)
    )
    print(df['question_word'].value_counts())
    sns.countplot(y='question_word', data=df, order=df['question_word'].value_counts().index)
    plt.title("Most Common Question Words")
    plt.xlabel("Count")
    plt.ylabel("Question Word")
    plt.show()
    return

if __name__ == '__main__':
    login(HF_TOKEN)
    requests = load_dataset(REQ_DS)
    df = requests['train'].to_pandas()
    calc_qw(df)