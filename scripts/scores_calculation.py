import argparse

from config import REFERENCES_CSV
from src.metric.scorer import ScoringPipeline


if __name__ == '__main__':
   parser = argparse.ArgumentParser(description="Run the scoring pipeline.")
   parser.add_argument('--version', type=str, required=True, help='Run version')
   args = parser.parse_args()
   
   scorer = ScoringPipeline(REFERENCES_CSV, args.version)
   scorer.exec_pipeline()
   scorer.save()