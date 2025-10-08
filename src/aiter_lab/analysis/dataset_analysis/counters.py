import numpy as np
import pandas as pd
from collections import Counter
from ...utils import clean_text
import ast

import nltk
from nltk.corpus import stopwords

def safe_nltk_download(resource_name, subdir):
    try:
        nltk.data.find(f'{subdir}/{resource_name}')
    except LookupError:
        nltk.download(resource_name)

safe_nltk_download('punkt', 'tokenizers')
safe_nltk_download('stopwords', 'corpora')

def words_counter(df):
    stop_words = set(stopwords.words('french'))
    def get_filtered_words_from_clean(msg):
        return {w for w in msg.split() if w not in stop_words and len(w) > 5}
    word_counts = Counter()
    df['clean_msg'] = df['opening_msg'].apply(clean_text)
    for msg in df['clean_msg']:
        filtered_words = get_filtered_words_from_clean(msg)
        word_counts.update(filtered_words)
    return word_counts

def convert_to_list(val):
    if isinstance(val, str):
        try:
            return ast.literal_eval(val)
        except:
            return [np.nan]
    return val

def list_features_counter(df, feature):
    col = df[feature].apply(convert_to_list)
    exploded = col.explode()
    exploded = exploded[~pd.isnull(exploded)]
    counts = Counter(exploded)
    return counts