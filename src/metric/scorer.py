import pandas as pd
from datetime import datetime
import os

from src.metric.reformulation import create_reformulations
from src.metric.ter_computation import compute_scores

from src.utils.utils import load_from_hf
from config import HYP_DS, METADATA_CSV, VERSION, REFORMULATION_MODEL, REFERENCES_CSV

csv_cols = ["conv_id", "model_id", "request_id", "request", "reference", "context", "hypothesis", "filtered_hypothesis", "corrected_hypothesis", "score", "ot_score", "global_score"]

class ScoringPipeline:
    def __init__(self, overwrite=None, steps=range(3)):
        self.version = VERSION
        self.overwrite = overwrite
        self.model = REFORMULATION_MODEL
        self.filepath = self._get_filepath()
        
        try:
            self.df = pd.read_csv(self.filepath)
        except FileNotFoundError:
            self.df = pd.DataFrame(columns=csv_cols)
        
        self.references_csv = REFERENCES_CSV
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
        if self.overwrite is not None:
            return self.overwrite
        i = 1
        while True:
            filepath = f'data/results/results_{i}.csv'
            if not os.path.exists(filepath):
                return filepath
            i += 1

    
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
            hf_new = hf_new.rename(columns={
                'response': 'hypothesis'
            })
            new_df = pd.merge(hf_new, self.references_df, on='request_id', how='inner', suffixes=("", "_dup"))
            new_df["filtered_hypothesis"] = None
            new_df["corrected_hypothesis"] = None
            new_df["score"] = None
            new_df["ot_score"] = None
            new_df["global_score"] = None
            new_df = new_df[csv_cols]
            self.df = pd.concat([self.df, new_df], ignore_index=True)
            
    def reformulation(self):
        self.df = create_reformulations(self.df, self.model)
        return
    
    def scoring(self):
        self.df = compute_scores(self.df)
        return
    
    def metadata_creation(self):
        metadata_row = {
            "filepath": self.filepath,
            "reformulation_col": "",
            "reformulation_model": self.model,
            "code_version": self.version["CODE_VERSION"],
            "prompt_version": self.version["PROMPT_VERSION"],
            "references_version": self.version["REFERENCES_VERSION"],
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