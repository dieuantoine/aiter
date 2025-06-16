import pandas as pd
from sacrebleu.metrics import TER

from tqdm import tqdm
tqdm.pandas()

df = pd.read_csv("resultats.csv")

ter = TER()

def compute_ter(ref, hyp):
    return ter.sentence_score(hyp, [ref]).score

df["TER"] = df.progress_apply(
    lambda row: compute_ter(row["reference"], row["reformulation"]),
    axis=1
)

df.to_csv("resultats_avec_ter.csv", index=False)