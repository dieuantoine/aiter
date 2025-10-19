import os
from pathlib import Path
from dotenv import load_dotenv
from .utils import load_yaml

load_dotenv()

# Chemin racine du projet
BASE_DIR = Path(__file__).resolve().parents[3]

# Secrets
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
HF_TOKEN = os.getenv("HF_TOKEN")

# Version du code
CONFIG_DIR = BASE_DIR / "config"
version_config = load_yaml(CONFIG_DIR / "version.yaml")
VERSION = version_config['VERSION']
LANG = VERSION["LANG"]
TESTED_MODELS = VERSION['TESTED_MODELS']
DATASET = VERSION['DATASET']

# Chemins importants

DATA_DIR = BASE_DIR / "data"

INTERIM_DIR = DATA_DIR / "interim" / VERSION["DATASET"]

SELECTED_IDS_CSV = INTERIM_DIR / f"selected_req_ids_{VERSION['DATA_VERSION']}_{LANG}.csv"
REFERENCES_CSV = INTERIM_DIR / f"references_{VERSION['DATA_VERSION']}_{LANG}.csv"

TEMP_HYP_CSV = INTERIM_DIR / f"hypotheses_{VERSION['DATA_VERSION']}_{LANG}.csv"

RESULTS_DIR = DATA_DIR / "results"

ANALYSIS_DIR = DATA_DIR / "analysis"

DS_ANALYSIS_DIR = ANALYSIS_DIR / "datasets_analysis"

RESULTS_ANALYSIS_DIR = ANALYSIS_DIR / "results_analysis"

METADATA_CSV = RESULTS_DIR / "metadata.csv"

#Datasets
ds_config = load_yaml(CONFIG_DIR / "hf_ds_paths.yaml")
ds = ds_config['datasets']
username = ds_config['username']
CONV_DS = ds['CONV_DS']
REAC_DS = ds['REAC_DS']
MKQA_DS = ds['MKQA_DS']
LOCAL_MKQA_PATH = DATA_DIR / "raw" / "mkqa.jsonl.gz"
REQ_DS = f'{username}/{DATASET}-requests-{LANG}'
HYP_DS = f'{username}/{DATASET}-hypotheses-{LANG}'
    
# Question Words
qw_config = load_yaml(CONFIG_DIR / "question_words.yaml")
question_words = qw_config['QUESTION_WORDS'][LANG]