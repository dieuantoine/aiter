from src.data.preprocessing import create_req_and_hyp_ds

from config import REAC_DS, CONV_DS, REQ_DS, HYP_DS

if __name__ == '__main__':
   create_req_and_hyp_ds(CONV_DS, REAC_DS, REQ_DS, HYP_DS)