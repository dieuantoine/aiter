from huggingface_hub import login
import matplotlib.pyplot as plt
import pandas as pd
from .counters import words_counter, list_features_counter
from .visualization import plot_msg_length, create_pie, compare_msg_length, compare_hyp_length
from .hypotheses_analysis import merge_stats_tab, len_correlations
from ...config import HF_TOKEN, DS_ANALYSIS_DIR, RESULTS_DIR, COMPARIA_REQ_DS, MKQA_REQ_DS, MKQA_HYP_DS, REFERENCES_CSV
from ...utils import load_from_hf

if __name__ == '__main__':
    df = pd.read_csv(RESULTS_DIR / 'results_1.csv')
    len_correlations(df, DS_ANALYSIS_DIR / 'correlation_matrix.png')