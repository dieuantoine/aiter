from huggingface_hub import login
import matplotlib.pyplot as plt
import pandas as pd
from src.analysis.dataset_analysis.counters import words_counter, list_features_counter
from src.analysis.dataset_analysis.visualization import plot_msg_length, create_pie, compare_msg_length
from config import HF_TOKEN, DS_ANALYSIS_DIR, COMPARIA_REQ_DS, MKQA_REQ_DS
from src.utils.utils import load_from_hf

if __name__ == '__main__':
    login(HF_TOKEN)
    df1 = load_from_hf(MKQA_REQ_DS)
    df2 = load_from_hf("dieuant/requests-dataset")
    compare_msg_length(df1, df2, ["MKQA", "compar:IA"], DS_ANALYSIS_DIR / "mkqa_comparia_msg_length_comparison.png")