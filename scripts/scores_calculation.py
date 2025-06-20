from config import REFERENCES_CSV
from src.metric.scorer import ScoringPipeline


if __name__ == '__main__':
   scorer = ScoringPipeline(REFERENCES_CSV, "test0")
   scorer.exec_pipeline()
   scorer.save()
   
   