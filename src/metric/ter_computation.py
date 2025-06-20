import pandas as pd
from sacrebleu.metrics import TER

from tqdm import tqdm
tqdm.pandas()

def compute_ter(ter, ref, hyp):
    return ter.sentence_score(hyp, [ref]).score

def compute_scores(df):
    ter = TER()
    mask = df['score'].isna()
    if mask.any():
        for idx in df[mask].index:
            df.at[idx, 'score'] = compute_ter(ter, df.at[idx, 'reference'], df.at[idx, 'reformulation'])

    return df