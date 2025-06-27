import pandas as pd
from sacrebleu.metrics import TER

from tqdm import tqdm

def compute_ter(ter, ref, hyp):
    return ter.sentence_score(hyp, [ref]).score

def compute_scores(df, reformulation_col):
    ter = TER()
    mask = df['score'].isna()
    if reformulation_col=="hyp":
        ref_col, hyp_col = 'response', 'reformulation'
    elif reformulation_col=="ref":
        ref_col, hyp_col = 'reformulation', 'response'
    if mask.any():
        for idx in tqdm(df[mask].index, desc="Calcul des scores TER"):
            df.at[idx, 'score'] = compute_ter(ter, df.at[idx, ref_col], df.at[idx, hyp_col])

    return df