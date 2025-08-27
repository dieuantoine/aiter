from src.llm_api.llm_api_call import create_client, call_api
from src.utils.utils import load_prompt
import pandas as pd
from tqdm import tqdm
from config import OT_PROMPT, COR_PROMPT, REF_PROMPT, COM_PROMPT

class Request:
    def __init__(self, req: str = "", ref: str = "", con: str = ""):
        self.req = req
        self.ref = ref
        self.con = con 

    @classmethod
    def from_series(cls, row: pd.Series):
        return cls(
            req=row.get("request", ""),
            ref=row.get("reference", ""),
            con=row.get("context", ""),
        )

def format_prompt(base_prompt: str, request: Request, hypothesis: str) -> str:
    return base_prompt.format(
        req=request.req,
        ref=request.ref,
        con=request.con,
        hyp=hypothesis
        )

def reformulate(client, base_prompt: str, reformulation_model: str, request: Request, hypothesis: str) -> str:
    prompt = format_prompt(base_prompt, request, hypothesis)
    return call_api(client, prompt, model=reformulation_model)

def create_reformulations(df, reformulation_model):
    mask = df['filtered_hypothesis'].isna()
    if mask.any():
        client = create_client(reformulation_model)
        ot_base_prompt = load_prompt(OT_PROMPT)
        cor_base_prompt = load_prompt(COR_PROMPT)
        com_base_prompt = load_prompt(COM_PROMPT)
        for idx in tqdm(df[mask].index, desc="Reformulation"):
            request = Request.from_series(df.loc[idx])
            hypothesis = df.loc[idx, 'hypothesis']
            filtered = reformulate(client, ot_base_prompt, reformulation_model, request, hypothesis)
            corrected = reformulate(client, cor_base_prompt, reformulation_model, request, filtered)
            completed = reformulate(client, com_base_prompt, reformulation_model, request, corrected)
            df.at[idx, 'filtered_hypothesis'] = filtered
            df.at[idx, 'corrected_hypothesis'] = completed
    return df

# def create_reformulations_1(df, reformulation_model):
#     mask = df['corrected_hypothesis'].isna()
#     if mask.any():
#         client = create_client()
#         base_prompt = load_prompt(REF_PROMPT)
#         for idx in tqdm(df[mask].index, desc="Reformulation"):
#             response = Response.from_series(df.loc[idx])
#             if response.is_valid():
#                 response.corrected = reformulate(client, base_prompt, reformulation_model, response)
#                 df.at[idx, 'corrected_hypothesis'] = response.corrected
#     return df

# def create_reformulations_2(df, reformulation_model):
#     mask = df['filtered_hypothesis'].isna()
#     if mask.any():
#         client = create_client()
#         ot_base_prompt = load_prompt(OT_PROMPT)
#         cor_base_prompt = load_prompt(COR_PROMPT)
#         for idx in tqdm(df[mask].index, desc="Off-topic filtering"):
#             response = Response.from_series(df.loc[idx])
#             if response.is_valid():
#                 response.filtered = reformulate(client, ot_base_prompt, reformulation_model, response)
#                 filtered_input = Response(
#                     req=response.req,
#                     ref=response.ref,
#                     con=response.con,
#                     hyp=response.filtered
#                 )
#                 response.corrected = reformulate(client, cor_base_prompt, reformulation_model, filtered_input)
#                 df.at[idx, 'filtered_hypothesis'] = response.filtered
#                 df.at[idx, 'corrected_hypothesis'] = response.corrected
#     return df