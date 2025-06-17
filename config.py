import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Chemin racine du projet
BASE_DIR = Path(__file__).resolve().parent

# Chemins importants

DATA_DIR = BASE_DIR / "data"

SELECTED_IDS_CSV = DATA_DIR / "selected_req_ids.csv"
REFERENCES_CSV = DATA_DIR / "references.csv"

PROMPTS_DIR = BASE_DIR / "src" / "metric" / "prompts"

REFORMULATION_PROMPT = PROMPTS_DIR / "reformulation_prompt.txt"
REFUSAL_CLASSIFIER_PROMPT = PROMPTS_DIR / "refusal_classifier_prompt.txt"

# Secrets
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
HF_TOKEN = os.getenv("HF_TOKEN")

#Huggingface datasets
CONV_DS = "ministere-culture/comparia-conversations"
REAC_DS = "ministere-culture/comparia-reactions"

REQ_DS = "dieuant/requests-dataset"
HYP_DS = "dieuant/hypothesis-dataset"

#French Question Words
question_words = ['quoi', 'quand', 'comment', 'pourquoi', 'où', 'qui', 'quel', 'quelle', 'quels', 'quelles', 'lequel', 'combien']
