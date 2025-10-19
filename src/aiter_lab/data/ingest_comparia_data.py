from datasets import load_dataset
from huggingface_hub import login

from ..config import HF_TOKEN

from .utils import calc_msg_length, calc_qw

def select_valid_msg(conv_dataset, reac_dataset):
    conv_valid_msg = set(
        x['opening_msg'] for x in conv_dataset 
        if x['languages'] == ['fr'] and len(x['opening_msg']) > 10
    )
    reac_valid_msg = set(
        x['opening_msg'] for x in reac_dataset
        if x['conv_turns'] == 1 and (
            x['useful'] or x['creative'] or x['complete'] or x['clear_formatting'] or x['incorrect'] or x['superficial'] or x['instructions_not_followed']
        )
    )
    valid_msg = conv_valid_msg & reac_valid_msg
    return valid_msg

def reduce_dataset(ds, columns, valid_msg):
    col_to_remove = [x for x in ds.column_names if x not in columns]
    return ds.filter(lambda x: x['opening_msg'] in valid_msg).remove_columns(col_to_remove)

def arrange_comparia_req(conv_dataset, valid_msg):
    req_col = ['opening_msg', 'categories']
    df = reduce_dataset(conv_dataset, req_col, valid_msg).to_pandas()
    df = df.drop_duplicates(subset='opening_msg')
    df['request_id'] = range(len(df))
    df = calc_msg_length(df, 'opening_msg')
    df = calc_qw(df, 'opening_msg')
    df = df.rename(columns={
        'opening_msg': 'request'
    })
    df = df[['request_id', 'request', 'categories', 'msg_length', 'question_words']]
    return df

def arrange_comparia_hyp(reac_dataset, valid_msg, req_df):
    hyp_col = ['refers_to_model', 'opening_msg', 'response_content', 'liked', 'disliked', 'useful', 'creative', 'complete', 'clear_formatting', 'incorrect', 'superficial', 'instructions_not_followed']
    df = reduce_dataset(reac_dataset, hyp_col, valid_msg).to_pandas()
    df = df.rename(columns={
        'opening_msg': 'request',
        'refers_to_model': 'model_id',
        'response_content': 'hypothesis'
    })
    request_id_map = req_df[['request_id', 'request']]
    df = df.merge(request_id_map, on='request', how='left')
    df['conv_id'] = range(len(df))
    df = df[['conv_id', 'model_id', 'request_id', 'request', 'hypothesis', 'liked', 'disliked', 'useful', 'creative', 'complete', 'clear_formatting', 'incorrect', 'superficial', 'instructions_not_followed']]
    return df

def create_comparia_df(conv_ds_path, reac_ds_path):
    login(HF_TOKEN)
    conv_dataset = load_dataset(conv_ds_path)['train']
    reac_dataset = load_dataset(reac_ds_path)['train']
    valid_msg = select_valid_msg(conv_dataset, reac_dataset)
    req_df = arrange_comparia_req(conv_dataset, valid_msg)
    hyp_df = arrange_comparia_hyp(reac_dataset, valid_msg, req_df)
    return req_df, hyp_df