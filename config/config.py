import os
from pathlib import Path
from dotenv import load_dotenv
from config.utils import load_yaml

load_dotenv()

# Chemin racine du projet
BASE_DIR = Path(__file__).resolve().parents[1]

# Secrets
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
HF_TOKEN = os.getenv("HF_TOKEN")

#Huggingface datasets
ds_config = load_yaml(BASE_DIR / "config" / "hf_ds_paths.yaml")
ds = ds_config['datasets']
CONV_DS = ds['CONV_DS']
REAC_DS = ds['REAC_DS']
REQ_DS = ds['REQ_DS']
HYP_DS = ds['HYP_DS']

#French Question Words
question_words = ['quoi', 'quand', 'comment', 'pourquoi', 'où', 'qui', 'quel', 'quelle', 'quels', 'quelles', 'lequel', 'combien']

# Version du code
version_config = load_yaml(BASE_DIR / "config" / "version.yaml")
VERSION = version_config['VERSION']

REFORMULATION_MODEL = "mistral-medium-latest"

# Chemins importants

DATA_DIR = BASE_DIR / "data"

SELECTED_IDS_CSV = DATA_DIR / "selected_req_ids.csv"
print(VERSION["REFERENCES_VERSION"])
REFERENCES_CSV = DATA_DIR / f"references_{VERSION['REFERENCES_VERSION']}.csv"

METADATA_CSV = DATA_DIR / "metadata.csv"

PROMPTS_DIR = BASE_DIR / "src" / "metric" / "prompts"

REF_REFORMULATION_PROMPT = PROMPTS_DIR / f"ref_reformulation_prompt_{VERSION['PROMPT_VERSION']}.txt"
HYP_REFORMULATION_PROMPT = PROMPTS_DIR / f"hyp_reformulation_prompt_{VERSION['PROMPT_VERSION']}.txt"
REFUSAL_CLASSIFIER_PROMPT = PROMPTS_DIR / "refusal_classifier_prompt.txt"