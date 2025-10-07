import time
from google import genai
from google.genai import types
from config import MODEL_CONFIG

def create_googleai_client():
    return genai.Client()

def call_googleai_api(client, prompt, model, temperature=0.0, call_delay=MODEL_CONFIG["call_delay"], retry_delay=MODEL_CONFIG["retry_delay"], max_retries=3):
    retries = 0
    while retries < max_retries:
        try:
            chat_response = client.models.generate_content(
                model=model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=temperature
                )
            )
            time.sleep(call_delay)
            return chat_response.text

        except AttributeError as e:
            print(f"Erreur d'attribut : {e}")
            return 

        except Exception as e:
            print(f"Une erreur est survenue : {e}")
            retries += 1
            if retries >= max_retries:
                print(f"Erreur après {max_retries} tentatives : {e}")
                return None
            print(f"Nouvelle tentative ({retries}/{max_retries}) dans {retry_delay} secondes...")
            time.sleep(retry_delay)