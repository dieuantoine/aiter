import pandas as pd
from datetime import datetime
import os

from aiter import Scorer

from ..utils import load_from_hf
from ..config import HYP_DS, VERSION, RESULTS_DIR, REFERENCES_CSV, METADATA_CSV, MISTRAL_API_KEY, GOOGLE_API_KEY

cols_base = ["conv_id", "model_id", "request_id", "request", "reference", "context", "hypothesis"]
cols = {"1": ["corrected_hypothesis", "score"], 
        "2": ["filtered_hypothesis", "corrected_hypothesis", "cor_score", "ot_score", "score"],
        "3": ["filtered_hypothesis", "revised_hypothesis", "corrected_hypothesis", "cor_score", "ot_score", "score"]
}

class ScoringPipeline:
    def __init__(self, result_filename=None):
        self.version = VERSION
        self.model = VERSION["REFORMULATION_MODEL"]
        self.filepath = self._get_filepath(result_filename)
        
        self.duration_time = None

        self.csv_cols = cols_base + cols[self.version["CODE_VERSION"]]
        self.result_df = pd.DataFrame(columns=self.csv_cols)
        
        self.references_df = pd.read_csv(REFERENCES_CSV)
        self.references_df['request_id'] = self.references_df['request_id'].astype(str)
        
    def _get_filepath(self, result_filename):
        if result_filename:
            return RESULTS_DIR / result_filename
        i = 1
        while True:
            filepath = RESULTS_DIR / f'results_{i}.csv'
            if not os.path.exists(filepath):
                return filepath
            i += 1

    def init_result_df(self):
        
        new_ids = set(self.references_df[self.references_df['reference_created'] == True]['request_id'])
        
        if len(new_ids)>0:
            hyp_df = load_from_hf(HYP_DS)
            hyp_df['request_id'] = hyp_df['request_id'].astype(str)
            hyp_new = hyp_df[hyp_df['request_id'].isin(new_ids)].copy()
            new_df = pd.merge(hyp_new, self.references_df, on='request_id', how='inner', suffixes=("", "_dup"))
            for col in cols[self.version["CODE_VERSION"]]:
                new_df[col] = None
            new_df = new_df[cols_base + cols[self.version["CODE_VERSION"]]]
            self.result_df = pd.concat([self.result_df, new_df], ignore_index=True)
            
    def scoring(self):
        start_time = datetime.now()
        scorer = Scorer(self.result_df, self.version)
        scorer.reformulation()
        scorer.scoring()
        self.result_df = scorer.df
        end_time = datetime.now()
        self.duration_time = (end_time - start_time).total_seconds()
        return

    def metadata_creation(self):
        metadata_row = {
            "filepath": self.filepath,
            "reformulation_model": self.model,
            "code_version": self.version["CODE_VERSION"],
            "dataset": self.version["DATASET"],
            "data_version": self.version["DATA_VERSION"],
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "duration": f"{self.duration_time}s"
        }
        
        file_exists = os.path.isfile(METADATA_CSV)
        metadata_df = pd.DataFrame([metadata_row])
    
        metadata_df.to_csv(METADATA_CSV, mode='a', index=False, header=not file_exists)
    
    def save(self):
        self.result_df[self.csv_cols].to_csv(self.filepath, index=False)
    
    def exec_pipeline(self):
        self.init_result_df()
        self.scoring()
        self.save()
        self.metadata_creation()

if __name__ == "__main__":
    pipeline = ScoringPipeline()
    pipeline.exec_pipeline()