import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Chemin racine du projet
BASE_DIR = Path(__file__).resolve().parent

# Chemins importants


# Secrets
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
HF_TOKEN = os.getenv("HF_TOKEN")

#Huggingface datasets
CONV_DS = "ministere-culture/comparia-conversations"
REAC_DS = "ministere-culture/comparia-reactions"

REQ_DS = "requests-dataset"
HYP_DS = "hypothesis-dataset"