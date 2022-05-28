import torch
import torch.nn as nn
import torch.nn.functional as F
from attentions import MultiHeadAttention
from positional_encoders import AbsolutePositionalEncoder, T5RelativePositionalEncoder



class TransformerEncoderBlock(nn.Module):
    def __init__(self, emb_dim, num_heads, positional_encoding="abs"):
        super(TransformerEncoderBlock, self).__init__()
        self.positional_encoding = positional_encoding

        self.encoder_layers = MultiHeadAttention(emb_dim, num_heads, positional_encoding)

        self.midLayerNorm = nn.LayerNorm(emb_dim, eps=1e-05)

        self.ff_layer = nn.Linear(emb_dim, emb_dim)

        self.outLayerNorm = nn.LayerNorm(emb_dim, eps=1e-05)

    def forward(self, input_embedding, relative_bias=None, is_dropout=True):
        query = key = value = input_embedding

        # multi-head-attention layer
        if self.positional_encoding == "t5":
            hidden_states, attn_score = self.encoder_layers(query, key, value, relative_bias=relative_bias, is_dropout=is_dropout)
        else:
            hidden_states, attn_score = self.encoder_layers(query, key, value, is_dropout=is_dropout)

        # add & norm layer
        hidden_states = self.midLayerNorm(input_embedding+hidden_states)

        # feed-forward layer
        ff_hidden_states = self.ff_layer(hidden_states)

        # add & norm layer
        output_hidden_states = self.outLayerNorm(hidden_states+ff_hidden_states)

        return output_hidden_states, attn_score


class TransformerEncoder(nn.Module):
    def __init__(self, emb_dim, num_heads, num_layers, positional_encoding="abs"):
        super(TransformerEncoder, self).__init__()
        self.positional_encoding = positional_encoding

        self.num_layers = num_layers

        self.encoder_blocks = nn.ModuleList([TransformerEncoderBlock(emb_dim, num_heads, positional_encoding) for x in range(num_layers)])

    def forward(self, input_embedding, relative_bias=None):
        output_list = []
        attn_score_list = []

        if self.positional_encoding == "t5":
            output, attn_score = self.encoder_blocks[0](input_embedding, relative_bias)
            output_list.append(output)
            attn_score_list.append(attn_score)

            for i in range(1, self.num_layers-1):
                output, attn_score = self.encoder_blocks[i](output, relative_bias)
                output_list.append(output)
                attn_score_list.append(attn_score)
            output, attn_score = self.encoder_blocks[self.num_layers-1](output, relative_bias, is_dropout=False)
            output_list.append(output)
            attn_score_list.append(attn_score)

        else:
            output, attn_score = self.encoder_blocks[0](input_embedding)
            output_list.append(output)
            attn_score_list.append(attn_score)

            for i in range(1, self.num_layers-1):
                output, attn_score = self.encoder_blocks[i](output)
                output_list.append(output)
                attn_score_list.append(attn_score)
            output, attn_score = self.encoder_blocks[self.num_layers-1](output, relative_bias, is_dropout=False)
            output_list.append(output)
            attn_score_list.append(attn_score)

        return output, output_list, attn_score_list



def get_candidate_heads(emb_dim, _num_heads):
    divisor_list = []
    for i in range(1, emb_dim):
        if emb_dim % i == 0:
            divisor_list.append(i)
    return divisor_list[len(divisor_list)//2]


def get_model(emb_dim, num_heads, num_layers, selected_pe):
    return TransformerEncoder(emb_dim, num_heads, num_layers, selected_pe)


def get_output(input_embedding, num_layers, selected_pe, _num_heads):
    emb_dim = input_embedding.size()[-1]
    
    # input embedding + positional encoding
    positional_encoder = AbsolutePositionalEncoder(emb_dim)
    input_embedding = input_embedding + positional_encoder(input_embedding)

    seq_len_query = seq_len_key = input_embedding.size()[1]

    if emb_dim % _num_heads != 0:
        num_heads = get_candidate_heads(emb_dim, _num_heads)
    else:
        num_heads = _num_heads


    model = get_model(emb_dim, num_heads, num_layers, selected_pe)

    if selected_pe == "t5":
        relative_position_bias = T5RelativePositionalEncoder(num_heads)
        relative_bias = relative_position_bias(seq_len_query, seq_len_key)
        output, output_list, attn_score_list = model(input_embedding, relative_bias=relative_bias)
    else:
        # Absolute Positional Encoding and Relative Positional Encoding
        output, output_list, attn_score_list = model(input_embedding)

    return model, output, output_list, attn_score_list
