import torch
import torch.nn as nn
import torch.nn.functional as F
from positional_encoders import RelativePositionalEncoder


# Scaled Dot Product Attention using Absolute Positional Encoding
class ScaledDotProductAttention(nn.Module):
    def __init__(self, emb_dim):
        super(ScaledDotProductAttention, self).__init__()

        # scaling factor 1 / sqrt(dimension of queries and keys)
        self.scaling_factor = torch.sqrt(torch.tensor(emb_dim))


    def forward(self, query, key, value, mask = None):
        # Scaled score of the Matrix multiplication of query and key (e)
        attn_score = torch.bmm(query, key.transpose(1, 2)) / self.scaling_factor

        # Masking (Optional)
        # shape of mask: (batch size, input length of query, input length of key)
        if mask is not None:
            attn_score.masked_fill_(mask, -1e18)

        # Softmax of the scaled score (alpha)
        attn_score = F.softmax(attn_score, -1)

        # Matrix multiplication of the scaled score and value (z)
        output = torch.bmm(attn_score, value)

        return output, attn_score


# Scaled Dot Product Attention using Relative Positional Encoding
class RelativeScaledDotProductAttention(nn.Module):
    def __init__(self, emb_dim):
        super(RelativeScaledDotProductAttention, self).__init__()

        # scaling factor 1 / sqrt(dimension of queries and keys)
        self.scaling_factor = torch.sqrt(torch.tensor(emb_dim))


    def forward(self, query, key, value, a_key, a_value, mask = None):

        # Scaled score of the Matrix multiplication of query and key (e)
        qk_attn = torch.bmm(query, key.transpose(1, 2))
        relative_qk_attn = torch.bmm(query.permute(1, 0, 2).contiguous(), a_key.transpose(1, 2)).transpose(0, 1)
        attn_score = (qk_attn + relative_qk_attn) / self.scaling_factor

        # Masking (Optional)
        # shape of mask: (batch size, input length of query, input length of key)
        if mask is not None:
            attn_score.masked_fill_(mask, -1e18)

        # Softmax of the scaled score (alpha)
        attn_score = F.softmax(attn_score, -1)

        # Matrix multiplication of the scaled score and value (z)
        qkv_attn = torch.bmm(attn_score, value)
        relative_qkv_attn = torch.bmm(attn_score.permute(1, 0, 2).contiguous(), a_value).transpose(0, 1)

        output = qkv_attn + relative_qkv_attn

        return output, attn_score



# Scaled Dot Product Attention using T5 Relative Positional Encoding
class T5ScaledDotProductAttention(nn.Module):
    def __init__(self, emb_dim):
        super(T5ScaledDotProductAttention, self).__init__()

        # scaling factor 1 / sqrt(dimension of queries and keys)
        self.scaling_factor = torch.sqrt(torch.tensor(emb_dim))


    def forward(self, query, key, value, relative_bias, mask = None):
        # Scaled score of the Matrix multiplication of query and key (e)
        attn_score = torch.bmm(query, key.transpose(1, 2)) / self.scaling_factor + relative_bias.permute(2,0,1)

        # Masking (Optional)
        # shape of mask: (batch size, input length of query, input length of key)
        if mask is not None:
            attn_score.masked_fill_(mask, -1e18)

        # Softmax of the scaled score (alpha)
        attn_score = F.softmax(attn_score, -1)

        output = torch.bmm(attn_score, value)

        return output, attn_score




# Multi-Head Attention using Relation Positional Encoding
class MultiHeadAttention(nn.Module):
    def __init__(self, emb_dim, num_heads, positional_encoding="abs", dropout_rate=0.1):
        super(MultiHeadAttention, self).__init__()

        self.head_dim = int(emb_dim / num_heads)
        self.num_heads = num_heads
        self.positional_encoding = positional_encoding
        self.dropout = nn.Dropout(p=dropout_rate)

        # initialize one feed-forward layer (head dimension x number of heads) of each q, k and v
        # instead of initializing number of heads of feed-forward layers (head dimension / number of heads)
        self.query_proj = nn.Linear(emb_dim, self.head_dim * num_heads)
        self.key_proj = nn.Linear(emb_dim, self.head_dim * num_heads)
        self.value_proj = nn.Linear(emb_dim, self.head_dim * num_heads)
        self.out_proj = nn.Linear(emb_dim, self.head_dim * num_heads)

        if positional_encoding == "abs":
            self.scaled_dot_attn = ScaledDotProductAttention(self.head_dim)

        elif positional_encoding == "rel":
            self.relative_scaled_dot_attn = RelativeScaledDotProductAttention(self.head_dim)
            self.relative_position_k = RelativePositionalEncoder(self.head_dim)
            self.relative_position_v = RelativePositionalEncoder(self.head_dim)

        elif positional_encoding == "t5":
            self.t5_scaled_dot_attn = T5ScaledDotProductAttention(self.head_dim)


    def reshape_from_feed_forward(self, batch_size, _tensor):
        return _tensor.view(batch_size, -1, self.num_heads, self.head_dim)


    def reshape_to_ScaledDotProductAttention(self, batch_size, _tensor):
        # before shape: (batch size, input length, number of heads, head dimension)
        # after shape: (batch size, number of heads, input length, head dimension)
        _tensor = _tensor.permute(0, 2, 1, 3)

        # reshape to feed the tensor to ScaledDotProductAttention
        return _tensor.contiguous().view(batch_size * self.num_heads, -1, self.head_dim)


    def reshape_to_concat(self, batch_size, _tensor):
        # before shape: (number of heads, batch size, input length, head dimension)
        # after shape: (batch size, input length, number of heads, head dimension)
        _tensor = _tensor.permute(1, 2, 0, 3)
        # return shape: (batch size, input length, number of heads * head dimension)
        return _tensor.contiguous().view(batch_size, -1, self.num_heads * self.head_dim)


    def forward(self, query, key, value, mask = None, relative_bias=None, is_dropout=True):
        # shape of input of q, k and v: (batch size, input length, embedding dimension)
        batch_size = query.size()[0]

        # feed-forward network
        query = self.query_proj(query)
        key = self.key_proj(key)
        value = self.value_proj(value)


        # reshape the result of the feed-forward network
        # shape after the feed-forward network of q, k and v: (batch, input length, number of heads, head dimension)
        query = self.reshape_from_feed_forward(batch_size, query)
        key = self.reshape_from_feed_forward(batch_size, key)
        value = self.reshape_from_feed_forward(batch_size, value)

        # reshape the result of the feed-forward network to feed it to ScaledDotProductAttention
        # shape: (number of heads * batch, input length, head dimension)
        query = self.reshape_to_ScaledDotProductAttention(batch_size, query)
        key = self.reshape_to_ScaledDotProductAttention(batch_size, key)
        value = self.reshape_to_ScaledDotProductAttention(batch_size, value)


        # shape of mask: (batch size, number of heads, input length of query, input length of key)
        if mask is not None:
            mask = mask.unsqueeze(1).repeat(1, self.num_heads, 1, 1)

        if self.positional_encoding == "abs":
            output, attn_score = self.scaled_dot_attn(query, key, value, mask)

        elif self.positional_encoding == "rel":
            seq_len_query = query.size()[1]
            seq_len_key = key.size()[1]
            seq_len_value = value.size()[1]
            a_key = self.relative_position_k(seq_len_query, seq_len_key)
            a_value = self.relative_position_v(seq_len_query, seq_len_value)
            output, attn_score = self.relative_scaled_dot_attn(query, key, value, a_key, a_value, mask)

        elif self.positional_encoding == "t5":
            output, attn_score = self.t5_scaled_dot_attn(query, key, value, relative_bias, mask)


        # reshape the result of the ScaledDotProductAttention
        # shape: (number of heads, batch size, input length, head dimension)
        output = output.view(self.num_heads, batch_size, -1, self.head_dim)

        # reshape to concat
        # shape: (number of heads, batch size, input length, head dimension)
        output = self.reshape_to_concat(batch_size, output)

        # final feed-forward network
        output = self.out_proj(output)

        if is_dropout:
            output = self.dropout(output)
            return output, attn_score

        return output, attn_score
