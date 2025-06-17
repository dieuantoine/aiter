import pandas as pd
from sacrebleu.metrics import TER

from tqdm import tqdm
tqdm.pandas()

def compute_ter(ter, ref, hyp):
    return ter.sentence_score(hyp, [ref]).score

def compute_scores(ref_csv):
    df = pd.read_csv(ref_csv)
    ter = TER()

    df["TER"] = df.progress_apply(
        lambda row: compute_ter(ter, row["reference"], row["reformulation"]),
        axis=1
    )
    
    df.to_csv(ref_csv, index=False)
    return