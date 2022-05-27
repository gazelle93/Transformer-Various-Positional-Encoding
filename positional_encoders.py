import torch
import torch.nn as nn


class AbsolutePositionalEncoder(nn.Module):
    def __init__(self, emb_dim, max_position=512):
        super(AbsolutePositionalEncoder, self).__init__()
        self.position = torch.arange(max_position).unsqueeze(1)

        self.positional_encoding = torch.zeros(1, max_position, emb_dim)

        _2i = torch.arange(0, emb_dim, step=2).float()

        # PE(pos, 2i) = sin(pos/10000^(2i/d_model))
        self.positional_encoding[0, :, 0::2] = torch.sin(self.position / (10000 ** (_2i / emb_dim)))

        # PE(pos, 2i+1) = cos(pos/10000^(2i/d_model))
        self.positional_encoding[0, :, 1::2] = torch.cos(self.position / (10000 ** (_2i / emb_dim)))

    def forward(self, x):
        # batch_size, input_len, embedding_dim
        batch_size, seq_len, _ = x.size()

        return self.positional_encoding[:batch_size, :seq_len, :]

# https://github.com/tensorflow/tensor2tensor
class RelativePositionalEncoder(nn.Module):
    def __init__(self, emb_dim, max_position=512):
        super(RelativePositionalEncoder, self).__init__()
        self.max_position = max_position
        self.embeddings_table = nn.Parameter(torch.Tensor(max_position * 2 + 1, emb_dim))
        nn.init.xavier_uniform_(self.embeddings_table)

    def forward(self, seq_len_q, seq_len_k):
        range_vec_q = torch.arange(seq_len_q)
        range_vec_k = torch.arange(seq_len_k)
        relative_matrix = range_vec_k[None, :] - range_vec_q[:, None]
        clipped_relative_matrix = torch.clamp(relative_matrix, -self.max_position, self.max_position)
        relative_position_matrix = clipped_relative_matrix + self.max_position
        embeddings = self.embeddings_table[relative_position_matrix]

        return embeddings


class T5RelativePositionalEncoder(nn.Module):
    def __init__(self, num_heads, max_position=512):
        super(T5RelativePositionalEncoder, self).__init__()
        self.max_position = max_position
        self.embeddings_table = nn.Embedding(max_position*max_position, num_heads)

    def forward(self, seq_len_q, seq_len_k):
        range_vec_q = torch.arange(seq_len_q)
        range_vec_k = torch.arange(seq_len_k)
        relative_position = range_vec_k[None, :] - range_vec_q[:, None]
        relative_position_clipped = torch.clamp(relative_position, -self.max_position, self.max_position)
        final_mat = relative_position_clipped + self.max_position
        embeddings = self.embeddings_table(final_mat)

        return embeddings
