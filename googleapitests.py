from src.process.metric.reformulation import create_reformulations
from src.utils.utils import load_from_hf
from config import HYP_DS, VERSION, REFORMULATION_MODEL, RESULTS_DIR, REFERENCES_CSV, MISTRAL_API_KEY
import pandas as pd


cols_base = ["conv_id", "model_id", "request_id", "request", "reference", "context", "hypothesis"]
cols_calc = ["filtered_hypothesis", "corrected_hypothesis", "cor_score", "ot_score", "score"]
cols = cols_base + cols_calc

references_df = pd.read_csv(REFERENCES_CSV)

ids = [7960403399608384663, 7818112641371223126, -1881286304870722165, 3309140698422618645, 846466415522403319]

hyp_df = load_from_hf(HYP_DS)
hf_new = hyp_df[hyp_df['request_id'].isin(ids)].copy()
new_df = pd.merge(hf_new, references_df, on='request_id', how='inner', suffixes=("", "_dup"))
for col in cols_calc:
    new_df[col] = None
new_df = new_df[cols]

create_reformulations(new_df, REFORMULATION_MODEL).to_csv("tests.csv")

# import requests

# url = "https://api.mistral.ai/v1/models"
# headers = {"Authorization": f"Bearer {MISTRAL_API_KEY}"}
# response = requests.get(url, headers=headers)

# print("Status code:", response.status_code)
# data = response.json()['data']
# for model in data:
#     print(model['id'])