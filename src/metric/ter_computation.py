import pandas as pd
from sacrebleu.metrics import TER

from tqdm import tqdm

def not_non_empty_str(obj):
    return not isinstance(obj, str) or len(obj)==0

def compute_ter(ter, ref, hyp):
    if not_non_empty_str(ref) or not_non_empty_str(hyp):
        return 100
    return ter.sentence_score(hyp, [ref]).score

def compute_scores(df, reformulation_col):
    ter = TER(no_punct=True)
    mask = df['score'].isna()
    if reformulation_col=="hyp":
        ref_col, hyp_col = 'response', 'reformulation'
    elif reformulation_col=="ref":
        ref_col, hyp_col = 'reformulation', 'response'
    if mask.any():
        for idx in tqdm(df[mask].index, desc="Calcul des scores TER"):
            df.at[idx, 'score'] = compute_ter(ter, df.at[idx, ref_col], df.at[idx, hyp_col])

    return df