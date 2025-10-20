import pandas as pd
from .counters import words_counter, list_features_counter
from .visualization import plot_msg_length, create_pie, compare_msg_length, compare_hyp_length, create_wc
from .hypotheses_analysis import len_correlations
from ...config import HF_TOKEN, DS_ANALYSIS_DIR, RESULTS_DIR, HYP_DS, REQ_DS, REFERENCES_CSV
from ...utils import load_from_hf

if __name__ == '__main__':
    df = load_from_hf(REQ_DS)
    count = words_counter(df, 'request')
    create_wc(count, DS_ANALYSIS_DIR / 'wordcloud_requests_mkqa.png')