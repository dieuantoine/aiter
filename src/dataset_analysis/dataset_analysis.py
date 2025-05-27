from huggingface_hub import login
from datasets import load_dataset
import pandas as pd
from src.dataset_analysis.counters import words_counter, list_features_counter
from src.dataset_analysis.visualization import plot_msg_length, create_pie
from config import HF_TOKEN, REQ_INFO_DS

if __name__ == '__main__':
    login(HF_TOKEN)
    df = load_dataset(REQ_INFO_DS)['train'].to_pandas()
    counts = list_features_counter(df, 'categories')
    create_pie(counts)