import pandas as pd

from .utils import calc_msg_length, calc_qw

from ..utils import load_from_hf

def arrange_mkqa_req(mkqa_df, lang):
    mkqa_df["request"] = mkqa_df['queries'].apply(lambda x: x[lang])
    mkqa_df["reference_annotation"] = mkqa_df['answers'].apply(lambda x: x[lang][0]["text"])
    mkqa_df = calc_msg_length(mkqa_df, 'request')
    mkqa_df = calc_qw(mkqa_df, 'request')
    mkqa_df['categories'] = [[] for _ in range(len(mkqa_df))]
    mkqa_df = mkqa_df.rename(columns={
        'example_id': 'request_id'
    })
    req_df = mkqa_df[['request_id', 'request', 'categories', 'reference_annotation', 'msg_length', 'question_words']]
    return req_df

def arrange_mkqa_hyp(req_df, models):
    rows = []
    for _, row in req_df.iterrows():
        for model in models:
            rows.append({
                'conv_id': "",
                'model_id': model,
                'request_id': str(row["request_id"]),
                'request': row['request'],
                'hypothesis': ""
            })
    hyp_df = pd.DataFrame(rows)
    hyp_df['conv_id'] = range(len(hyp_df))
    return hyp_df

def create_mkqa_df(mkqa_ds_path, models, lang):
    mkqa_df = load_from_hf(mkqa_ds_path)
    req_df = arrange_mkqa_req(mkqa_df, lang)
    hyp_df = arrange_mkqa_hyp(req_df, models)
    return req_df, hyp_df