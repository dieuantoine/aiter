from datasets import load_dataset, Dataset
from huggingface_hub import login
from config import HF_TOKEN, CONV_DS, REAC_DS, REQ_DS, REQ_INFO_DS, HYP_DS, question_words

def select_valid_ids(conv_dataset, reac_dataset):
    conv_valid_ids = set(
        x['conversation_pair_id'] for x in conv_dataset 
        if not x['is_unedited_prompt'] and x['languages'] == ['fr'] and len(x['opening_msg'])>10
        )

    reac_valid_ids = set(
        x['conversation_pair_id'] for x in reac_dataset
        if x['conv_turns'] == 1 and (
            x['useful'] or x['creative'] or x['complete'] or x['clear_formatting'] or x['incorrect'] or x['superficial'] or x['instructions_not_followed']
            )
        )

    valid_ids = conv_valid_ids & reac_valid_ids
    
    return valid_ids

def reduce_dataset(ds, columns, valid_ids):
    col_to_remove = [x for x in ds.column_names if x not in columns]
    return ds.filter(lambda x: x['conversation_pair_id'] in valid_ids).remove_columns(col_to_remove)

def create_req_and_hyp_ds():
    conv_dataset = load_dataset(CONV_DS)['train']
    reac_dataset = load_dataset(REAC_DS)['train']
    valid_ids = select_valid_ids(conv_dataset, reac_dataset)
    req_col = ['conversation_pair_id', 'opening_msg', 'categories']
    hyp_col = ['conversation_pair_id', 'refers_to_model', 'refers_to_conv_id', 'question_content', 'response_content', 'liked', 'disliked', 'useful', 'creative', 'complete', 'clear_formatting', 'incorrect', 'superficial', 'instructions_not_followed']
    reduced_req_dataset = reduce_dataset(conv_dataset, req_col, valid_ids)
    reduced_hyp_dataset = reduce_dataset(reac_dataset, hyp_col, valid_ids)
    reduced_req_dataset.push_to_hub(REQ_DS)
    reduced_hyp_dataset.push_to_hub(HYP_DS)
    return

def calc_msg_length(df):
    df['msg_length'] = df['opening_msg'].apply(lambda x: len(str(x).split()))
    df['char_length'] = df['opening_msg'].apply(lambda x: len(str(x)))
    return df

def calc_qw(df):
    df['question_word'] = df['opening_msg'].apply(
        lambda x: list({w for w in question_words if w in x.lower().split()})
    )
    return df

def create_req_info_ds():
    df = load_dataset(REQ_DS)['train'].to_pandas()
    df = calc_msg_length(df)
    df = calc_qw(df)
    dataset = Dataset.from_pandas(df)
    dataset.push_to_hub(REQ_INFO_DS)
    return
    
if __name__ == '__main__':
    login(HF_TOKEN)
    create_req_info_ds()