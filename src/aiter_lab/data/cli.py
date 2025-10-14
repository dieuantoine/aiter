import argparse
import pandas as pd

from .ingest_comparia_data import create_comparia_df
from .ingest_mkqa_data import create_mkqa_df
from .hypotheses_creation import init_temp_hyp_csv, push_temp_hyp_csv

from ..utils import push_to_hf

from ..config import CONV_DS, REAC_DS, LOCAL_MKQA_PATH, HYP_DS, REQ_DS, TESTED_MODELS, DATASET, LANG, SELECTED_IDS_CSV, TEMP_HYP_CSV

def create_req_and_hyp_ds():
    if DATASET == "comparia" and LANG == "fr":
        req_df, hyp_df = create_comparia_df(CONV_DS, REAC_DS)
    elif DATASET == "mkqa":
        req_df, hyp_df = create_mkqa_df(LOCAL_MKQA_PATH, TESTED_MODELS, LANG)
    else:
        req_df, hyp_df = None, None
    if req_df is not None and hyp_df is not None:
        push_to_hf(req_df, REQ_DS)
        print(f"Requests dataset pushed to {REQ_DS}")
        push_to_hf(hyp_df, HYP_DS)
        print(f"Hypotheses dataset pushed to {HYP_DS}")
    else:
        print("No dataset created. Check the dataset version and language.")
    return

def create_temp_hyp_csv():
    if DATASET == "mkqa":
        init_temp_hyp_csv(HYP_DS, SELECTED_IDS_CSV, TEMP_HYP_CSV)
    else:
        print("Temporary hypotheses CSV is only implemented for the MKQA dataset.")
    return

def push_hyp_to_hf():
    if DATASET == "mkqa":
        push_temp_hyp_csv(HYP_DS, TEMP_HYP_CSV)
    else:
        print("Temporary hypotheses CSV is only implemented for the MKQA dataset.")
    return

def main():
    parser = argparse.ArgumentParser(description="CLI for datasets creation")
    parser.add_argument("action", choices=["create_ds", "create_temp_csv", "push_temp_csv"], help="Action to perform")
    args = parser.parse_args()
    if args.action == "create_ds":
        create_req_and_hyp_ds()
    elif args.action == "create_temp_csv":
        create_temp_hyp_csv()
    elif args.action == "push_temp_csv":
        push_temp_hyp_csv()
    else:
        print("Invalid action. Choose from 'create_ds', 'create_temp_csv', or 'push_temp_csv'.")

if __name__ == "__main__":
    main()