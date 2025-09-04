import pandas as pd
from datetime import datetime
import os

from src.process.metric.reformulation import create_reformulations
from src.process.metric.ter_computation import compute_scores

from src.utils.utils import load_from_hf
from config import HYP_DS, VERSION, REFORMULATION_MODEL, RESULTS_DIR, REFERENCES_CSV, METADATA_CSV

cols_base = ["conv_id", "model_id", "request_id", "request", "reference", "context", "hypothesis"]
cols = {"1": ["corrected_hypothesis", "score"], 
        "2": ["filtered_hypothesis", "corrected_hypothesis", "cor_score", "ot_score", "score"]}

class ScoringPipeline:
    def __init__(self, overwrite=None, steps=range(3)):
        self.version = VERSION
        self.overwrite = overwrite
        self.model = REFORMULATION_MODEL
        self.filepath = self._get_filepath()
        
        self.start_time = None

        self.csv_cols = cols_base + cols[self.version["CODE_VERSION"]]
        
        try:
            self.df = pd.read_csv(self.filepath)
        except FileNotFoundError:
            self.df = pd.DataFrame(columns=self.csv_cols)
        
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
            filepath = RESULTS_DIR / f'results_{i}.csv'
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
            new_df = pd.merge(hf_new, self.references_df, on='request_id', how='inner', suffixes=("", "_dup"))
            for col in cols[self.version["CODE_VERSION"]]:
                new_df[col] = None
            new_df = new_df[cols_base + cols[self.version["CODE_VERSION"]]]
            self.df = pd.concat([self.df, new_df], ignore_index=True)
            
    def reformulation(self):
        # if self.version["CODE_VERSION"] == "1":
        #     self.df = create_reformulations_1(self.df, self.model)
        # elif self.version["CODE_VERSION"] == "2":
        #     self.df = create_reformulations(self.df, self.model)
        self.df = create_reformulations(self.df, self.model)
        return
    
    def scoring(self):
        self.df = compute_scores(self.df, method=self.version["CODE_VERSION"])
        return
    
    def metadata_creation(self):
        end_time = datetime.now()
        duration = (end_time - self.start_time).total_seconds()
        metadata_row = {
            "filepath": self.filepath,
            "reformulation_model": self.model,
            "code_version": self.version["CODE_VERSION"],
            "prompt_version": self.version["PROMPT_VERSION"],
            "references_version": self.version["REFERENCES_VERSION"],
            "timestamp": self.start_time.strftime("%Y-%m-%d %H:%M:%S"),
            "duration": f"{duration} seconds",
            "new_ids_count": len(self.new_ids),
            "new_ids": ";".join(str(i) for i in self.new_ids)
        }
        
        file_exists = os.path.isfile(METADATA_CSV)
        metadata_df = pd.DataFrame([metadata_row])
    
        metadata_df.to_csv(METADATA_CSV, mode='a', index=False, header=not file_exists)
    
    def save(self):
        self.df[self.csv_cols].to_csv(self.filepath, index=False)
        self.metadata_creation()
    
    def exec_pipeline(self):
        self.start_time = datetime.now()
        for i in self.steps:
            step = self.pipeline[i]
            print(f"Step {i}: {step.__name__}")
            step()