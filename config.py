import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Chemin racine du projet
BASE_DIR = Path(__file__).resolve().parent

# Chemins importants


# Secrets
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
