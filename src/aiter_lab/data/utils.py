from ..config import question_words

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