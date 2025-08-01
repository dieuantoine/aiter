import pandas as pd
from evaluate import load
from tqdm import tqdm

def calculate_bertscore(df):
    bertscore = load("bertscore")
    recall_ref = []
    precision_ctx = []

    for _, row in tqdm(df.iterrows(), total=len(df)):
        hyp = row["hypothesis"]
        ref = row["reference"]
        ctx = row["context"]

        score_ref = bertscore.compute(
            predictions=[hyp],
            references=[ref],
            lang="fr"
        )
        recall_ref.append(score_ref["recall"][0])

        score_ctx = bertscore.compute(
            predictions=[hyp],
            references=[ctx],
            lang="fr"
        )
        precision_ctx.append(score_ctx["precision"][0])

    df["bertscore_recall_ref"] = recall_ref
    df["bertscore_precision_ctx"] = precision_ctx
    return df