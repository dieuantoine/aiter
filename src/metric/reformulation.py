from src.llm_api.mistral_api import call_mistral_api, create_client
from src.llm_api.mistral_batch_api import create_input_file, run_batch_job, download_file
from src.utils.utils import load_prompt
import pandas as pd
from tqdm import tqdm
from config import OFF_TOPIC_FILTERING_PROMPT, HYP_REFORMULATION_PROMPT

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
            request=row.get("request_id"),
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
    return call_mistral_api(client, prompt, model=reformulation_model, call_delay=3.0)

def create_reformulations(df, reformulation_model):
    mask = df['reformulation'].isna()
    if mask.any():
        client = create_client()
        ot_base_prompt = load_prompt(OFF_TOPIC_FILTERING_PROMPT)
        cor_base_prompt = load_prompt(HYP_REFORMULATION_PROMPT)
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