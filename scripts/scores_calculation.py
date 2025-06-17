from config import REFERENCES_CSV
from src.metric.reformulation import create_reformulations

if __name__ == '__main__':
   create_reformulations(REFERENCES_CSV)