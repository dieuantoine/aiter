from src.llm_api.mistral_api import call_mistral_api, create_client
from src.llm_api.mistral_batch_api import create_input_file, run_batch_job, download_file
from src.utils.utils import load_prompt
import pandas as pd
from tqdm import tqdm
from config import OT_PROMPT, COR_PROMPT, REF_PROMPT, COM_PROMPT

class Response:
    def __init__(self, conv_id: str = "", model: str = "", req_id: str = "", req: str = "", ref: str = "", con: str = "", hyp: str = ""):
        self.conv_id = conv_id
        self.model = model
        self.req_id = req_id
        self.req = req
        self.ref = ref
        self.con = con
        self.hyp = hyp
        self.filtered = None
        self.corrected = None

    def is_valid(self) -> bool:
        return bool(self.ref and self.hyp)

    def to_dict(self) -> dict:
        return {
            "model": self.model,
            "request": self.req,
            "reference": self.ref,
            "context": self.con,
            "hypothesis": self.hyp,
            "filtered_response": self.filtered,
            "corrected_response": self.corrected
        }

    @classmethod
    def from_series(cls, row: pd.Series):
        return cls(
            conv_id=row.get("conv_id"),
            model=row.get("model_id"),
            req_id=row.get("request_id"),
            req=row.get("request", ""),
            ref=row.get("reference", ""),
            con=row.get("context", ""),
            hyp=row.get("hypothesis", "")
        )

def format_prompt(base_prompt: str, response: Response) -> str:
    return base_prompt.format(
        req=response.req,
        ref=response.ref,
        con=response.con,
        hyp=response.hyp
    )

def reformulate(client, base_prompt: str, reformulation_model: str, response: Response) -> str:
    prompt = format_prompt(base_prompt, response)
    return call_mistral_api(client, prompt, model=reformulation_model, call_delay=2.0)

def create_reformulations(df, reformulation_model):
    mask = df['filtered_hypothesis'].isna()
    if mask.any():
        client = create_client()
        ot_base_prompt = load_prompt(OT_PROMPT)
        cor_base_prompt = load_prompt(COR_PROMPT)
        com_base_prompt = load_prompt(COM_PROMPT)
        for idx in tqdm(df[mask].index, desc="Off-topic filtering"):
            response = Response.from_series(df.loc[idx])
            if response.is_valid():
                response.filtered = reformulate(client, ot_base_prompt, reformulation_model, response)
                filtered_input = Response(
                    req=response.req,
                    ref=response.ref,
                    con=response.con,
                    hyp=response.filtered
                )
                correction = reformulate(client, cor_base_prompt, reformulation_model, filtered_input)
                corrected_input = Response(
                    req=response.req,
                    ref=response.ref,
                    con=response.con,
                    hyp = correction
                )
                response.corrected = reformulate(client, com_base_prompt, reformulation_model, corrected_input)
                df.at[idx, 'filtered_hypothesis'] = response.filtered
                df.at[idx, 'corrected_hypothesis'] = response.corrected
    return df


def create_reformulations_1(df, reformulation_model):
    mask = df['corrected_hypothesis'].isna()
    if mask.any():
        client = create_client()
        base_prompt = load_prompt(REF_PROMPT)
        for idx in tqdm(df[mask].index, desc="Reformulation"):
            response = Response.from_series(df.loc[idx])
            if response.is_valid():
                response.corrected = reformulate(client, base_prompt, reformulation_model, response)
                df.at[idx, 'corrected_hypothesis'] = response.corrected
    return df

def create_reformulations_2(df, reformulation_model):
    mask = df['filtered_hypothesis'].isna()
    if mask.any():
        client = create_client()
        ot_base_prompt = load_prompt(OT_PROMPT)
        cor_base_prompt = load_prompt(COR_PROMPT)
        for idx in tqdm(df[mask].index, desc="Off-topic filtering"):
            response = Response.from_series(df.loc[idx])
            if response.is_valid():
                response.filtered = reformulate(client, ot_base_prompt, reformulation_model, response)
                filtered_input = Response(
                    req=response.req,
                    ref=response.ref,
                    con=response.con,
                    hyp=response.filtered
                )
                response.corrected = reformulate(client, cor_base_prompt, reformulation_model, filtered_input)
                df.at[idx, 'filtered_hypothesis'] = response.filtered
                df.at[idx, 'corrected_hypothesis'] = response.corrected
    return df