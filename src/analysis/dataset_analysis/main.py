from huggingface_hub import login
import matplotlib.pyplot as plt
import pandas as pd
from src.analysis.dataset_analysis.counters import words_counter, list_features_counter
from src.analysis.dataset_analysis.visualization import plot_msg_length, create_pie, compare_msg_length, compare_hyp_length
from src.analysis.dataset_analysis.hypotheses_analysis import merge_stats_tab
from config import HF_TOKEN, DS_ANALYSIS_DIR, RESULTS_DIR, COMPARIA_REQ_DS, MKQA_REQ_DS, MKQA_HYP_DS, REFERENCES_CSV
from src.utils.utils import load_from_hf

if __name__ == '__main__':
    merge_stats_tab(DS_ANALYSIS_DIR / 'all_stats.csv')