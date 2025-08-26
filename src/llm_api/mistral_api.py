import time
from mistralai import Mistral
from config import MISTRAL_API_KEY

def create_client():
    return Mistral(api_key=MISTRAL_API_KEY)

def call_mistral_api(client, prompt, model, temperature=0.0, call_delay=1.0, retry_delay=5.0, max_retries=3):
    retries = 0
    while retries < max_retries:
        try:
            chat_response = client.chat.complete(
                model=model,
                temperature=temperature,
                messages=[
                    {
                        "role": "user",
                        "content": prompt,
                    },
                ]
            )
            time.sleep(call_delay)
            return chat_response.choices[0].message.content

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

        