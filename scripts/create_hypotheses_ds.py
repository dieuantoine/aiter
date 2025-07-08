import argparse

from src.data.hypotheses_creation import create_temp_hyp_csv, push_hyp_csv

from config import MKQA_REQ_DS, MKQA_HYP_DS, SELECTED_IDS_CSV, TEMP_HYP_CSV

def main(action):
    if action == "c":
        create_temp_hyp_csv(MKQA_REQ_DS, SELECTED_IDS_CSV, TEMP_HYP_CSV)
    elif action == "p":
        push_hyp_csv(MKQA_HYP_DS, TEMP_HYP_CSV)
    return

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run the datasets creation.")
    parser.add_argument('--action', type=str, required=True)
    args = parser.parse_args()
    main(args.action)