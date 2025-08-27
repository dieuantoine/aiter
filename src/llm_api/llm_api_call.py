from src.llm_api.googleai_api import create_googleai_client, call_googleai_api
from src.llm_api.mistral_api import create_mistral_client, call_mistral_api

def create_client(model):
    api_name = model.split("-")[0].lower()
    if api_name == "gemini":
        return create_googleai_client()
    elif api_name == "mistral":
        return create_mistral_client()
    else:
        raise ValueError(f"Unknown API name: {api_name}")


def call_api(client, prompt, model, **kwargs):
    api_name = model.split("-")[0].lower()
    if api_name == "gemini":
        return call_googleai_api(client, prompt, model, **kwargs)
    elif api_name == "mistral":
        return call_mistral_api(client, prompt, model, **kwargs)
    else:
        raise ValueError(f"Unknown API name: {api_name}")