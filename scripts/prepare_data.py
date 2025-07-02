from src.data.preprocessing import create_req_and_hyp_ds, create_mkqa_ds

from config import REAC_DS, CONV_DS, REQ_DS, HYP_DS, MKQA_DS, MKQA_REQ_DS

if __name__ == '__main__':
   #create_req_and_hyp_ds(CONV_DS, REAC_DS, REQ_DS, HYP_DS)
   create_mkqa_ds(MKQA_REQ_DS)