from ingest_comparia_data import create_comparia_df
from ingest_mkqa_data import create_mkqa_df
from hypotheses_creation import init_temp_hyp_csv
import pandas as pd

from ..utils import push_to_hf

from ..config import CONV_DS, REAC_DS, MKQA_DS, HYP_DS, REQ_DS, TESTED_MODELS, DATASET, LANG, SELECTED_IDS_CSV, TEMP_HYP_CSV

def create_req_and_hyp_ds():
    if DATASET == "comparia" and LANG == "fr":
        req_df, hyp_df = create_comparia_df(CONV_DS, REAC_DS)
    elif DATASET == "mkqa":
        req_df, hyp_df = create_mkqa_df(MKQA_DS, TESTED_MODELS, LANG)
    else:
        req_df, hyp_df = None, None
    if req_df is not None and hyp_df is not None:
        push_to_hf(req_df, REQ_DS)
        push_to_hf(hyp_df, HYP_DS)
    else:
        print("No dataset created. Check the dataset version and language.")
    return

def create_temp_hyp_csv():
    if DATASET == "mkqa":
        init_temp_hyp_csv(HYP_DS, SELECTED_IDS_CSV, TEMP_HYP_CSV)
    else:
        print("Temporary hypotheses CSV is only implemented for the MKQA dataset.")
    return

def push_temp_hyp_csv():
    if DATASET == "mkqa":
        temp_hyp_df = pd.read_csv(TEMP_HYP_CSV)
        push_to_hf(temp_hyp_df, HYP_DS)
    else:
        print("Temporary hypotheses CSV is only implemented for the MKQA dataset.")
    return