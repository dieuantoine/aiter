import re

def clean_text(text):
    text = text.lower()
    text = re.sub(r"[éèê]", "e", text)
    text = re.sub(r"[ç]", "c", text)
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    return text