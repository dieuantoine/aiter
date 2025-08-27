from src.llm_api.googleai_api import create_googleai_client, call_googleai_api
from src.llm_api.mistral_api import create_mistral_client, call_mistral_api
from config import MODEL_CONFIG

API_CLIENTS = {
    "gemini": create_googleai_client,
    "mistral": create_mistral_client
}

API_CALLS = {
    "gemini": call_googleai_api,
    "mistral": call_mistral_api
}

def create_client():
    api_name = MODEL_CONFIG["api"]
    if api_name in API_CLIENTS:
        return API_CLIENTS[api_name]()
    else:
        raise ValueError(f"Unknown API name: {api_name}")


def call_api(client, prompt, model, **kwargs):
    config = MODEL_CONFIG
    api_name = config["api"]
    if api_name in API_CALLS:
        return API_CALLS[api_name](client, prompt, model, **kwargs)
    else:
        raise ValueError(f"Unknown API name: {api_name}")