import pandas as pd
from evaluate import load
from tqdm import tqdm
import sys, os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from config import DATA_DIR

bertscore = load("bertscore")

filepath = DATA_DIR / "results/results_1.csv"

df = pd.read_csv(filepath)

recall_ref = []
f1_ref = []
precision_ctx = []
f1_ctx = []

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
    f1_ref.append(score_ref["f1"][0])

    score_ctx = bertscore.compute(
        predictions=[hyp],
        references=[ctx],
        lang="fr"
    )
    precision_ctx.append(score_ctx["precision"][0])
    f1_ctx.append(score_ctx["f1"][0])

df["bertscore_recall_ref"] = recall_ref
df["bertscore_f1_ref"] = f1_ref
df["bertscore_precision_ctx"] = precision_ctx
df["bertscore_f1_ctx"] = f1_ctx

df.to_csv("output.csv", index=False)