import yaml
from src.utils.utils import load_prompt
from config import MISTRAL_API_KEY, REFORMULATION_MODEL
from src.llm_api.mistral_api import call_mistral_api
from mistralai import Mistral
import csv

with open("tests/data.yaml", "r", encoding="utf-8") as f:
    data = yaml.safe_load(f)

requests_list = data["requests"]

def format_prompt(base_prompt, req, ref, hyp, con):
    return base_prompt.format(ref=ref, hyp=hyp, req=req, context=con)

base_ot_prompt = load_prompt("tests/off_topic.txt")
base_reformulation_prompt = load_prompt("tests/reformulation.txt")

csv_file = "tests/results.csv"
fieldnames = ["Request", "Reference", "Hypothesis", "Context", "Filtered Hypothesis", "Response"]

if __name__ == "__main__":
    

    # Write header only once
    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

    for i, request in enumerate(requests_list):
        req = request["request"]
        ref = request["reference"]
        hyp = request["response"]
        con = request["context"]

        if not ref or not hyp:
            continue

        client = Mistral(api_key=MISTRAL_API_KEY)

        ot_prompt = format_prompt(base_ot_prompt, req, ref, hyp, con)
        new_hyp = call_mistral_api(client, ot_prompt, model=REFORMULATION_MODEL, call_delay=3.0)
        ref_prompt = format_prompt(base_reformulation_prompt, req, ref, new_hyp, con)
        response = call_mistral_api(client, ref_prompt, model=REFORMULATION_MODEL, call_delay=3.0)

        if i==0:
            print(ot_prompt)

        # Save to CSV
        with open(csv_file, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writerow({
                "Request": req,
                "Reference": ref,
                "Hypothesis": hyp,
                "Context": con,
                "Filtered Hypothesis": new_hyp,
                "Response": response
            })
