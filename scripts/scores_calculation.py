import argparse
from src.process.metric.scorer import ScoringPipeline

if __name__ == '__main__':
   parser = argparse.ArgumentParser(description="Run the scoring pipeline.")
   parser.add_argument('--overwrite', type=str, required=False)
   args = parser.parse_args()
   
   scorer = ScoringPipeline(overwrite=args.overwrite)
   scorer.exec_pipeline()