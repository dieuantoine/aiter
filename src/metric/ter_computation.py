import pandas as pd
from sacrebleu.metrics import TER

from tqdm import tqdm

def compute_ter(ter, ref, hyp):
    return ter.sentence_score(hyp, [ref]).score

def compute_scores(df):
    ter = TER(no_punct=True)
    mask = df['score'].isna() & df['corrected_hypothesis'].notna()
    if mask.any():
        for idx in tqdm(df[mask].index, desc="Calcul des scores TER"):
            df.at[idx, 'score'] = compute_ter(ter, df.at[idx, 'corrected_hypothesis'], df.at[idx, 'filtered_hypothesis'])
            df.at[idx, 'ot_score'] = compute_ter(ter, df.at[idx, 'filtered_hypothesis'], df.at[idx, 'hypothesis'])
            df.at[idx, 'global_score'] = compute_ter(ter, df.at[idx, 'corrected_hypothesis'], df.at[idx, 'hypothesis'])
    return df