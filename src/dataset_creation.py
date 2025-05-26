from datasets import load_dataset
from huggingface_hub import login
from config import HF_TOKEN, CONV_DS, REAC_DS, REQ_DS, HYP_DS

def select_valid_ids(conv_dataset, reac_dataset):
    conv_valid_ids = set(
        x['id'] for x in conv_dataset 
        if not x['is_unedited_prompt'] and x['languages'] == ['fr'] and len(x['opening_msg'])>10
        )

    reac_valid_ids = set(
        x['id'] for x in reac_dataset
        if x['conv_turns'] == 1 and (
            x['useful'] or x['creative'] or x['complete'] or x['clear_formatting'] or x['incorrect'] or x['superficial'] or x['instructions_not_followed']
            )
        )

    valid_ids = conv_valid_ids & reac_valid_ids
    
    return valid_ids

def reduce_dataset(ds, columns, valid_ids):
    col_to_remove = [x for x in ds.column_names if x not in columns]
    return ds.filter(lambda x: x['id'] in valid_ids).remove_columns(col_to_remove)

if __name__ == '__main__':
    login(HF_TOKEN)
    conv_dataset = load_dataset(CONV_DS)['train']
    reac_dataset = load_dataset(REAC_DS)['train']
    valid_ids = select_valid_ids(conv_dataset, reac_dataset)
    req_col = ['conversation_pair_id', 'opening_msg', 'categories']
    hyp_col = ['conversation_pair_id', 'refers_to_model', 'refers_to_conv_id', 'question_content', 'response_content', 'liked', 'disliked', 'useful', 'creative', 'complete', 'clear_formatting', 'incorrect', 'superficial', 'instructions_not_followed']
    reduced_req_dataset = reduce_dataset(conv_dataset, req_col, valid_ids)
    reduced_hyp_dataset = reduce_dataset(reac_dataset, hyp_col, valid_ids)
    reduced_req_dataset.push_to_hub(REQ_DS)
    reduced_hyp_dataset.push_to_hub(HYP_DS)
    