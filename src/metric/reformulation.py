from src.llm_api.mistral_api import call_mistral_api
from src.utils.utils import load_from_hf, load_prompt
from mistralai import Mistral
import pandas as pd
from tqdm import tqdm
from config import HYP_DS, REFORMULATION_PROMPT, MISTRAL_API_KEY

model = "mistral-small-latest"

def format_prompt(base_prompt, ref, hyp):
    return base_prompt.format(ref=ref, hyp=hyp)

def reformulate(client, base_prompt, ref, hyp):
    prompt = format_prompt(base_prompt, ref, hyp)
    return call_mistral_api(client, prompt, model=model)

def create_reformulations(df):
    mask = df['reformulation'].isna()
        
    if mask.any():
        base_prompt = load_prompt(REFORMULATION_PROMPT)
        client = Mistral(api_key=MISTRAL_API_KEY)
        for idx in tqdm(df[mask].index, desc="Calcul des reformulations"):
            ref, hyp = df.at[idx, 'reference'], df.at[idx, 'response']
            df.at[idx, 'reformulation'] = reformulate(client, base_prompt, ref, hyp)

    return df