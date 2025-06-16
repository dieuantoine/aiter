from config import MISTRAL_API_KEY, HYP_DS, HF_TOKEN

from mistralai import Mistral

from huggingface_hub import login
from datasets import load_dataset

import pandas as pd

import time
from tqdm import tqdm
tqdm.pandas()

model = "mistral-small-latest"

def create_message(prompt, ref, hyp):
    message_content = f"""
        Ta tâche est de reformuler une réponse produite par un modèle de langage.
        Objectif : faire en sorte que la forme de la réponse corresponde à celle d'une réponse de référence, tout en conservant uniquement les faits présents dans la réponse d’origine.
        Instructions :
        1. Reformule la réponse candidate pour que sa structure, son style, sa syntaxe et sa présentation soient similaires à la réponse de référence.
        2. Ne supprime, modifie ni n’invente aucun fait : la reformulation doit refléter exactement le contenu factuel de la réponse candidate.
        3. Si des éléments d'information de la réponse candidate n'apparaissent pas dans la référence, réécris les tels quels.
        4. Ignore les éléments de style de la candidate : politesse, justification, incertitude, tournures vagues, etc.
        5. Ne change pas l’ordre des faits à moins que ce soit nécessaire pour coller à la forme de la référence.
        6. N’inclus aucun élément qui apparaît dans la réponse de référence mais qui n’est pas présent dans la réponse candidate.
        
        ---

        Réponse de référence (style à imiter) :
        {ref}

        Réponse candidate (fond à conserver) :
        {hyp}
    
        ---

        Réponse reformulée (mêmes faits que la candidate, même style que la référence) :"""
    return message_content

def calc_reformulation(client, message_content):
    time.sleep(5)
    chat_response = client.chat.complete(
        model = model,
        temperature = 0,
        messages = [
            {
                "role": "user",
                "content": message_content,
            },
        ]
    )
    return chat_response.choices[0].message.content

def refusal_classifier(client, hyp):
    message_content = f"""
        Tu joues le rôle d'un classificateur.

        Analyse la réponse suivante et détermine à quelle catégorie elle appartient :

        1. Ne répond pas et admet qu’il ne peut pas répondre.
        2. Répond, mais émet des réserves ou des incertitudes.
        3. Répond avec assurance, sans réserve.

        Donne uniquement le chiffre correspondant à la bonne catégorie (1, 2 ou 3).

        ---

        Réponse à analyser :
        {hyp}

        ---

        Catégorie :"""
    chat_response = client.chat.complete(
        model = model,
        temperature = 0,
        messages = [
            {
                "role": "user",
                "content": message_content,
            },
        ]
    )
    return chat_response.choices[0].message.content

def main():
    ref_df = pd.read_csv('tests/references.csv', delimiter=",")
    ref_df = ref_df[ref_df["reference"] != "X"]
    valid_conv_ids = set(ref_df["conversation_pair_id"])
    
    login(HF_TOKEN)
    hyp_df = load_dataset(HYP_DS)['train'].to_pandas()
    df = hyp_df[hyp_df["conversation_pair_id"].isin(valid_conv_ids)]
    merged_df = df.merge(ref_df, on="conversation_pair_id", how="inner")
    
    client = Mistral(api_key=MISTRAL_API_KEY)
    
    merged_df["reformulation_prompt"] = merged_df.apply(lambda row: create_message("", row["reference"], row["response_content"]), axis=1)
    merged_df["reformulation"] = merged_df.progress_apply(lambda row: calc_reformulation(client, row["reformulation_prompt"]), axis=1)
    merged_df.to_csv("resultats.csv", index=False)

    # message_content = create_message("",ref,hyp)
    # print(calc_reformulation(client, message_content))
    return
  
if __name__ == '__main__':
    main()