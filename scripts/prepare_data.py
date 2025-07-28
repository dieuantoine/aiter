import argparse

from src.process.data.preprocessing import create_req_and_hyp_ds, create_mkqa_ds

from config import VERSION, REAC_DS, CONV_DS, COMPARIA_REQ_DS, COMPARIA_HYP_DS, MKQA_DS, MKQA_REQ_DS

def main(dataset_version, lang):
   if dataset_version == "comparia":
      create_req_and_hyp_ds(CONV_DS, REAC_DS, COMPARIA_REQ_DS, COMPARIA_HYP_DS)
   elif dataset_version == "mkqa":
      create_mkqa_ds(MKQA_DS, MKQA_REQ_DS, lang)
   return

if __name__ == '__main__':
   parser = argparse.ArgumentParser(description="Run the datasets creation.")
   parser.add_argument('--lang', type=str, required=True)
   args = parser.parse_args()
   
   main(VERSION['DATASET_VERSION'], args.lang)