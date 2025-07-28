from huggingface_hub import login
from datasets import load_dataset
import matplotlib.pyplot as plt
import pandas as pd
# from analysis.counters import words_counter, list_features_counter
# from analysis.visualization import plot_msg_length, create_pie
from config import HF_TOKEN, CONV_DS
from src.utils.utils import load_from_hf

def print_unedited_prompts():
    df = load_from_hf(CONV_DS)
    filtered_df = df[df["is_unedited_prompt"] == True]
    opening_msgs = set(filtered_df["opening_msg"])
    print(opening_msgs)
    return

if __name__ == '__main__':
    login(HF_TOKEN)
    df = load_from_hf(CONV_DS)
    #counts = list_features_counter(df, 'categories')
    #create_pie(counts)