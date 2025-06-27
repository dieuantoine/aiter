import pandas as pd
from datetime import datetime
import os

from src.metric.reformulation import create_reformulations, create_batch_reformulation
from src.metric.ter_computation import compute_scores

from src.utils.utils import load_from_hf
from config import HYP_DS, METADATA_CSV

csv_cols = ["conv_id", "model_id", "request_id", "request", "reference", "response", "reformulation", "score"]

class ScoringPipeline:
    def __init__(self, references_csv, reformulation_col, version, steps=range(3)):
        self.version = version
        self.reformulation_col = reformulation_col
        self.filepath = self._get_filepath()
        
        try:
            self.df = pd.read_csv(self.filepath)
        except FileNotFoundError:
            self.df = pd.DataFrame(columns=csv_cols)
        
        self.references_csv = references_csv
        self.references_df = pd.read_csv(self.references_csv)
        
        self.new_ids = None
        self._synchronise_ids()
        
        self.pipeline = [
            self.reformulation,
            self.scoring,
            self.save
        ]
        self.steps = steps
        
    def _get_filepath(self):
        return f'data/results_{self.reformulation_col}_{self.version}.csv'
    
    def _synchronise_ids(self):
        present_ids = set(self.df['request_id']) if 'request_id' in self.df.columns else set()
        new_ids = set(self.references_df[
            (~self.references_df['request_id'].isin(present_ids)) &
            (self.references_df['reference_created'] == True)
        ]['request_id'])
        
        if len(new_ids)>0:
            self.new_ids = new_ids
            hyp_df = load_from_hf(HYP_DS)
            hf_new = hyp_df[hyp_df['request_id'].isin(new_ids)].copy()
            new_df = pd.merge(hf_new, self.references_df, on='request_id', how='inner', suffixes=("", "_dup"))
            new_df['reformulation'] = None
            new_df['score'] = None
            new_df = new_df[csv_cols]
            self.df = pd.concat([self.df, new_df], ignore_index=True)
            
    def reformulation(self):
        self.df = create_reformulations(self.df, self.reformulation_col)
        return
    
    def scoring(self):
        self.df = compute_scores(self.df, self.reformulation_col)
        return
    
    def metadata_creation(self):
        metadata_row = {
            "filepath": self.filepath,
            "reformulation_col": self.reformulation_col,
            "version": self.version,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "new_ids_count": len(self.new_ids),
            "new_ids": ";".join(str(i) for i in self.new_ids)
        }
        
        file_exists = os.path.isfile(METADATA_CSV)
        metadata_df = pd.DataFrame([metadata_row])
    
        metadata_df.to_csv(METADATA_CSV, mode='a', index=False, header=not file_exists)

    
    def save(self):
        self.df[csv_cols].to_csv(self.filepath, index=False)
        self.metadata_creation()
    
    def exec_pipeline(self):
        for i in self.steps:
            step = self.pipeline[i]
            print(f"Etape {step.__name__}")
            step()