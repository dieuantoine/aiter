from config import REFERENCES_CSV
from src.metric.reformulation import create_reformulations
from src.metric.scorer import compute_scores

if __name__ == '__main__':
   create_reformulations(REFERENCES_CSV)
   compute_scores(REFERENCES_CSV)