import torch
from sklearn import preprocessing
from text_processing import get_nlp_pipeline, word_tokenization

def get_tokens(text, _pipeline):
    selected_nlp_pipeline = get_nlp_pipeline(_pipeline)

    return word_tokenization(text, selected_nlp_pipeline, _pipeline)

def init_token2idx(text_list, _pipeline):
    selected_nlp_pipeline = get_nlp_pipeline(_pipeline)
    tk_lists = []
    for text in text_list:
        tk_lists.append(word_tokenization(text, selected_nlp_pipeline, _pipeline))
        
    whole_tokens = [x for tk_list in tk_lists for x in tk_list]
    set_of_tokens = list(set(whole_tokens)) + ["UNK"]
    
    set_of_tokens.sort()
    
    token2idx_dict = {}
    idx2token_dict = {}
    for idx, t in enumerate(set_of_tokens):
        token2idx_dict[t] = idx
        idx2token_dict[idx] = t
        
    return token2idx_dict, idx2token_dict

def tk2idx(_text, _pipeline, token2idx_dict, unk_ignore):
    selected_nlp_pipeline = get_nlp_pipeline(_pipeline)
    tk_lists = word_tokenization(_text, selected_nlp_pipeline, _pipeline)
        
    
    idx_list = []
    if unk_ignore == True:
        for tk in tk_lists:
            if tk in token2idx_dict:
                idx_list.append(token2idx_dict[tk])
            else:
                idx_list.append(token2idx_dict["UNK"])
    else:
        for tk in tk_lists:
            idx_list.append(token2idx_dict[tk])
    return idx_list

def custom_one_hot_encoding(_idx_list, dim):
    tensor_list = []
    
    for idx in _idx_list:
        temp = torch.zeros(dim)
        temp[idx] = 1
        tensor_list.append(temp)
        
    return tensor_list

def tensor2token(_tensor, idx2token_dict):
    idx = (_tensor == 1).nonzero(as_tuple=True)[0].item()
    return idx2token_dict[idx]

def build_onehot_encoding_model(unk_ignore):
    if unk_ignore == True:
        model = preprocessing.OneHotEncoder(handle_unknown='ignore')
    else:
        model = preprocessing.OneHotEncoder()

    return model

def get_onehot_encoding(text_list, cur_text, _nlp_pipeline, _unk_igsnore):
    sklearn_onehotencoder = build_onehot_encoding_model(_unk_igsnore)
    token2idx_dict, _ = init_token2idx(text_list, _nlp_pipeline)
    sklearn_onehotencoder.fit([[t] for t in token2idx_dict])

    _tks = get_tokens(cur_text, _nlp_pipeline)

    tk_list = [[x] for x in _tks]

    return torch.tensor(sklearn_onehotencoder.transform(tk_list).toarray(), dtype=torch.float)
