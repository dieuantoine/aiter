import argparse
from src.metric.scorer import ScoringPipeline

if __name__ == '__main__':
   parser = argparse.ArgumentParser(description="Run the scoring pipeline.")
   parser.add_argument('--overwrite', type=str, required=False)
   parser.add_argument('--col', type=str, required=True)
   args = parser.parse_args()
   
   scorer = ScoringPipeline(args.col, overwrite=args.overwrite)
   scorer.exec_pipeline()