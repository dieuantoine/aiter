import pandas as pd

from src.metric.reformulation import create_reformulations
from src.metric.ter_computation import compute_scores

from utils.utils import load_from_hf
from config import REFERENCES_CSV, HYP_DS

class ScoringPipeline:
    def __init__(self, references_csv, version):
        self.references_csv = references_csv
        filepath = f'data/results_{version}'
        self.filepath = filepath
        
        try:
            self.df = pd.read_csv(self.filepath)
        except FileNotFoundError:
            self.df = pd.DataFrame(columns=["conv_id", "model_id", "request_id", "request", "reference", "hypothesis"])
        
        self.references_df = pd.read_csv(REFERENCES_CSV)
        
        self._synchronise_ids()
        
        self.pipeline = [
            self.reformulation,
            self.scoring
        ]
    
    def _synchronise_ids(self):
        present_ids = set(self.df['request_id']) if 'request_id' in self.df.columns else set()
        new_ids = self.references_df[
            (~self.references_df['request_id'].isin(present_ids)) &
            (self.references_df['reference_created'] == True)
        ].copy()
        
        if not new_ids.empty:
            hyp_df = load_from_hf(HYP_DS)
            hf_new = hyp_df[hyp_df['request_id'].isin(new_ids)].copy()
            new_df = pd.merge(hf_new, self.references_df, on='request_id', how='inner')
            new_df['reformulation'] = None
            new_df['score'] = None
            
    def reformulation(self):
        self.df = create_reformulations(self.df)
        return
    
    def scoring(self):
        self.df = compute_scores(self.df)
        return
    
    def exec_pipeline(self):
        for step in self.pipeline:
            step()
    
    def save(self):
        self.df.to_csv(self.filepath, index=False)