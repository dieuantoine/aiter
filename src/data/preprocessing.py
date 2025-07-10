from datasets import load_dataset, Dataset
from huggingface_hub import login

from config import HF_TOKEN, question_words

### COMPARIA 

def select_valid_msg(conv_dataset, reac_dataset):
    conv_valid_msg = set(
        x['opening_msg'] for x in conv_dataset 
        if not x['is_unedited_prompt'] and x['languages'] == ['fr'] and len(x['opening_msg'])>10
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

def calc_msg_length(df, col):
    df['msg_length'] = df[col].apply(
        lambda x: len(str(x).split())
    )
    return df

def calc_qw(df, col):
    df['question_words'] = df[col].apply(
        lambda x: list({w for w in question_words if w in x.lower().split()})
    )
    return df

def arrange_req_ds(conv_dataset, valid_msg):
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
    req_ds = Dataset.from_pandas(df)
    return req_ds

def arrange_hyp_ds(reac_dataset, valid_msg, req_ds):
    req_df = req_ds.to_pandas()
    hyp_col = ['refers_to_model', 'opening_msg', 'response_content', 'liked', 'disliked', 'useful', 'creative', 'complete', 'clear_formatting', 'incorrect', 'superficial', 'instructions_not_followed']
    df = reduce_dataset(reac_dataset, hyp_col, valid_msg).to_pandas()
    df = df.rename(columns={
        'opening_msg': 'request',
        'refers_to_model': 'model_id',
        'response_content': 'response'
    })
    request_id_map = req_df[['request_id', 'request']]
    df = df.merge(request_id_map, on='request', how='left')
    df['conv_id'] = range(len(df))
    df = df[['conv_id', 'model_id', 'request_id', 'request', 'response', 'liked', 'disliked', 'useful', 'creative', 'complete', 'clear_formatting', 'incorrect', 'superficial', 'instructions_not_followed']]
    hyp_ds = Dataset.from_pandas(df)
    return hyp_ds

def create_req_and_hyp_ds(conv_ds_path, reac_ds_path, req_ds_path, hyp_ds_path):
    login(HF_TOKEN)
    conv_dataset = load_dataset(conv_ds_path)['train']
    reac_dataset = load_dataset(reac_ds_path)['train']
    valid_msg = select_valid_msg(conv_dataset, reac_dataset)
    req_ds = arrange_req_ds(conv_dataset, valid_msg)
    hyp_ds = arrange_hyp_ds(reac_dataset, valid_msg, req_ds)
    req_ds.push_to_hub(req_ds_path)
    hyp_ds.push_to_hub(hyp_ds_path)
    return

### MKQA

def create_mkqa_ds(mkqa_ds_path, mkqa_req_ds_path, lang):
    login(HF_TOKEN)
    mkqa_df = load_dataset(mkqa_ds_path)['train'].to_pandas()
    mkqa_df["request"] = mkqa_df['queries'].apply(lambda x: x[lang])
    mkqa_df["reference_annotation"] = mkqa_df['answers'].apply(lambda x: x[lang][0]["text"])
    mkqa_df = calc_msg_length(mkqa_df, 'request')
    mkqa_df = calc_qw(mkqa_df, 'request')
    mkqa_df['categories'] = [[] for _ in range(len(mkqa_df))]
    mkqa_df = mkqa_df.rename(columns={
        'example_id': 'request_id'
    })
    mkqa_df = mkqa_df[['request_id', 'request', 'categories', 'reference_annotation', 'msg_length', 'question_words']]
    mkqa_ds = Dataset.from_pandas(mkqa_df)
    mkqa_ds.push_to_hub(mkqa_req_ds_path)
    return