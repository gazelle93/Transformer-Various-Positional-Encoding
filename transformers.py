import warnings

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

    def forward(self, input_embedding, relative_bias=None, mask=None, is_dropout=True):
        query = key = value = input_embedding

        # multi-head-attention layer
        hidden_states, attn_score = self.encoder_layers(
            query, key, value, mask=mask, relative_bias=relative_bias,
            is_dropout=is_dropout)

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

        # T5's bias is shared across layers, so it lives on the encoder rather
        # than in any block. Holding it here also puts it in .parameters(),
        # which a module built outside the model would never reach.
        self.relative_position_bias = (
            T5RelativePositionalEncoder(num_heads) if positional_encoding == "t5"
            else None)

    def forward(self, input_embedding, relative_bias=None, mask=None):
        output_list = []
        attn_score_list = []

        if self.positional_encoding == "t5" and relative_bias is None:
            seq_len = input_embedding.size(1)
            relative_bias = self.relative_position_bias(seq_len, seq_len)
        elif self.positional_encoding != "t5":
            relative_bias = None

        output = input_embedding
        for i, block in enumerate(self.encoder_blocks):
            # dropout is skipped on the final block only
            is_dropout = i < self.num_layers - 1
            output, attn_score = block(output, relative_bias=relative_bias,
                                       mask=mask, is_dropout=is_dropout)
            output_list.append(output)
            attn_score_list.append(attn_score)

        return output, output_list, attn_score_list


def get_candidate_heads(emb_dim, num_heads):
    """Nearest head count that divides emb_dim.

    Returns num_heads untouched when it already divides emb_dim. Otherwise it
    picks the closest divisor and says so, because silently changing a model's
    head count is expensive to debug.
    """
    if emb_dim % num_heads == 0:
        return num_heads

    divisors = [i for i in range(1, emb_dim + 1) if emb_dim % i == 0]
    # ties go to the larger divisor, so you never get less parallelism than asked
    nearest = min(divisors, key=lambda d: (abs(d - num_heads), -d))
    warnings.warn(
        f"num_heads={num_heads} does not divide emb_dim={emb_dim}; "
        f"using num_heads={nearest} instead.", stacklevel=2)
    return nearest


def get_model(emb_dim, num_heads, num_layers, selected_pe):
    return TransformerEncoder(emb_dim, num_heads, num_layers, selected_pe)


def get_output(input_embedding, num_layers, selected_pe, _num_heads, mask=None):
    emb_dim = input_embedding.size()[-1]

    # Absolute encoding is added at the input, and only when it is the scheme
    # being used. Shaw et al. and Raffel et al. both replace it rather than
    # stack on top of it, so adding it unconditionally would make every "rel"
    # or "t5" run a mixture and the comparison meaningless.
    if selected_pe == "abs":
        positional_encoder = AbsolutePositionalEncoder(emb_dim)
        input_embedding = input_embedding + positional_encoder(input_embedding)

    num_heads = get_candidate_heads(emb_dim, _num_heads)

    model = get_model(emb_dim, num_heads, num_layers, selected_pe)

    output, output_list, attn_score_list = model(input_embedding, mask=mask)

    return model, output, output_list, attn_score_list
