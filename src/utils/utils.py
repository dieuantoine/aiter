from datasets import load_dataset, Dataset
from huggingface_hub import login
from config import HF_TOKEN

import re

def load_from_hf(path):
    login(HF_TOKEN)
    return load_dataset(path)['train'].to_pandas()

def clean_text(text):
    text = text.lower()
    text = re.sub(r"[éèê]", "e", text)
    text = re.sub(r"[ç]", "c", text)
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    return text