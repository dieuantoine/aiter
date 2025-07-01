from src.llm_api.mistral_api import call_mistral_api
from src.llm_api.mistral_batch_api import create_input_file, run_batch_job, download_file
from src.utils.utils import load_from_hf, load_prompt
from mistralai import Mistral
import pandas as pd
from tqdm import tqdm
from config import HYP_DS, REF_REFORMULATION_PROMPT, HYP_REFORMULATION_PROMPT, MISTRAL_API_KEY

def format_prompt(base_prompt, ref, hyp):
    return base_prompt.format(ref=ref, hyp=hyp)

def reformulate(client, base_prompt, ref, hyp, reformulation_model):
    prompt = format_prompt(base_prompt, ref, hyp)
    return call_mistral_api(client, prompt, model=reformulation_model, call_delay=3.0)

def create_reformulations(df, reformulation_col, reformulation_model):
    
    if reformulation_col=="hyp":
        reformulation_prompt = HYP_REFORMULATION_PROMPT
    elif reformulation_col=="ref":
        reformulation_prompt = REF_REFORMULATION_PROMPT
    
    mask = df['reformulation'].isna()
        
    if mask.any():
        base_prompt = load_prompt(reformulation_prompt)
        client = Mistral(api_key=MISTRAL_API_KEY)
        for idx in tqdm(df[mask].index, desc="Calcul des reformulations"):
            ref, hyp = df.at[idx, "reference"], df.at[idx, "response"]
            df.at[idx, 'reformulation'] = reformulate(client, base_prompt, ref, hyp, reformulation_model)

    return df

def create_batch_reformulation(df, reformulation_col, reformulation_model):
    #Pas possible avec le free trial
    if reformulation_col=="hyp":
        reformulation_prompt = HYP_REFORMULATION_PROMPT
    elif reformulation_col=="ref":
        reformulation_prompt = REF_REFORMULATION_PROMPT
    base_prompt = load_prompt(reformulation_prompt)
    prompts = []
    for idx in tqdm(df.index, desc="Calcul des reformulations"):
        ref, hyp = df.at[idx, "reference"], df.at[idx, "response"]
        prompts.append(format_prompt(base_prompt, ref, hyp))
    client = Mistral(api_key=MISTRAL_API_KEY)
    input_file = create_input_file(client, prompts)
    print(f"Created input file {input_file}")

    batch_job = run_batch_job(client, input_file, reformulation_model)
    print(f"Job duration: {batch_job.completed_at - batch_job.created_at} seconds")
    download_file(client, batch_job.error_file, "error.jsonl")
    download_file(client, batch_job.output_file, "output.jsonl")