from src.llm_api.mistral_api import call_mistral_api
from src.utils.utils import load_from_hf, load_prompt
from mistralai import Mistral
import pandas as pd
from config import HYP_DS, REFORMULATION_PROMPT, MISTRAL_API_KEY

model = "mistral-small-latest"

def format_prompt(prompt, ref, hyp):
    return prompt.format(ref=ref, hyp=hyp)

def reformulate(client, prompt, ref, hyp):
    prompt = format_prompt(ref, hyp)
    return call_mistral_api(client, prompt, model=model)

def create_reformulations(ref_csv):
    ref_df = pd.read_csv(ref_csv, delimiter=",")
    ref_df = ref_df[ref_df["reference"] != "X"]
    valid_conv_ids = set(ref_df["request_id"])
    hyp_df = load_from_hf(HYP_DS)
    df = hyp_df[hyp_df["request_id"].isin(valid_conv_ids)]
    merged_df = df.merge(ref_df, on="request_id", how="inner")
    
    prompt = load_prompt(REFORMULATION_PROMPT)
    
    client = Mistral(api_key=MISTRAL_API_KEY)
    
    merged_df["reformulation"] = merged_df.progress_apply(lambda row: reformulate(client, prompt, row['reference'], row['response']), axis=1)
    merged_df.to_csv(ref_csv, index=False)
    return