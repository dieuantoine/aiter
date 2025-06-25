import os
from pathlib import Path
from dotenv import load_dotenv
import yaml

load_dotenv()

# Chemin racine du projet
BASE_DIR = Path(__file__).resolve().parents[1]

# Chemins importants

DATA_DIR = BASE_DIR / "data"

SELECTED_IDS_CSV = DATA_DIR / "selected_req_ids.csv"
REFERENCES_CSV = DATA_DIR / "references.csv"

PROMPTS_DIR = BASE_DIR / "src" / "metric" / "prompts"

REF_REFORMULATION_PROMPT = PROMPTS_DIR / "ref_reformulation_prompt.txt"
HYP_REFORMULATION_PROMPT = PROMPTS_DIR / "hyp_reformulation_prompt.txt"
REFUSAL_CLASSIFIER_PROMPT = PROMPTS_DIR / "refusal_classifier_prompt.txt"

# Secrets
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
HF_TOKEN = os.getenv("HF_TOKEN")

#Huggingface datasets
with open(BASE_DIR / "config" / "hf_ds_paths.yaml", "r") as f:
    config = yaml.safe_load(f)

ds = config['datasets']
CONV_DS = ds['CONV_DS']
REAC_DS = ds['REAC_DS']
REQ_DS = ds['REQ_DS']
HYP_DS = ds['HYP_DS']

#French Question Words
question_words = ['quoi', 'quand', 'comment', 'pourquoi', 'où', 'qui', 'quel', 'quelle', 'quels', 'quelles', 'lequel', 'combien']
