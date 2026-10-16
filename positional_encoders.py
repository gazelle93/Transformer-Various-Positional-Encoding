import math

import torch
import torch.nn as nn


class AbsolutePositionalEncoder(nn.Module):
    """Sinusoidal encoding from Vaswani et al. (2017), section 3.5."""

    def __init__(self, emb_dim, max_position=512):
        super(AbsolutePositionalEncoder, self).__init__()
        if emb_dim % 2 != 0:
            raise ValueError(
                f"emb_dim must be even for sinusoidal encoding, got {emb_dim}: "
                "the sin and cos halves would differ in length")

        position = torch.arange(max_position).unsqueeze(1)
        positional_encoding = torch.zeros(1, max_position, emb_dim)
        _2i = torch.arange(0, emb_dim, step=2).float()

        # PE(pos, 2i) = sin(pos/10000^(2i/d_model))
        positional_encoding[0, :, 0::2] = torch.sin(position / (10000 ** (_2i / emb_dim)))

        # PE(pos, 2i+1) = cos(pos/10000^(2i/d_model))
        positional_encoding[0, :, 1::2] = torch.cos(position / (10000 ** (_2i / emb_dim)))

        # a buffer, not a plain attribute: it has to follow .to(device) and
        # appear in state_dict, neither of which happens for a bare tensor
        self.register_buffer("positional_encoding", positional_encoding)

    def forward(self, x):
        # batch_size, input_len, embedding_dim
        _, seq_len, _ = x.size()
        if seq_len > self.positional_encoding.size(1):
            raise ValueError(
                f"sequence length {seq_len} exceeds max_position "
                f"{self.positional_encoding.size(1)}")
        # leading dim stays 1 and broadcasts over the batch
        return self.positional_encoding[:, :seq_len, :]


# https://github.com/tensorflow/tensor2tensor
class RelativePositionalEncoder(nn.Module):
    """Clipped relative position embeddings from Shaw et al. (2018).

    max_position is the clipping distance k: everything further apart than k
    shares an embedding. Shaw et al. found small k (around 16) sufficient.
    """

    def __init__(self, emb_dim, max_position=512):
        super(RelativePositionalEncoder, self).__init__()
        self.max_position = max_position
        self.embeddings_table = nn.Parameter(torch.empty(max_position * 2 + 1, emb_dim))
        nn.init.xavier_uniform_(self.embeddings_table)

    def forward(self, seq_len_q, seq_len_k):
        device = self.embeddings_table.device
        range_vec_q = torch.arange(seq_len_q, device=device)
        range_vec_k = torch.arange(seq_len_k, device=device)
        relative_matrix = range_vec_k[None, :] - range_vec_q[:, None]
        clipped_relative_matrix = torch.clamp(relative_matrix, -self.max_position, self.max_position)
        relative_position_matrix = clipped_relative_matrix + self.max_position
        embeddings = self.embeddings_table[relative_position_matrix]

        return embeddings


class T5RelativePositionalEncoder(nn.Module):
    """Relative position bias from Raffel et al. (2019), section 2.1.

    T5 does not embed every distance. It maps each one into a small number of
    buckets, exact for near positions and logarithmically widening for far
    ones, and learns one scalar per bucket per head. The bias is added to the
    attention logits and shared across layers.
    """

    def __init__(self, num_heads, num_buckets=32, max_distance=128,
                 bidirectional=True):
        super(T5RelativePositionalEncoder, self).__init__()
        self.num_buckets = num_buckets
        self.max_distance = max_distance
        self.bidirectional = bidirectional
        self.embeddings_table = nn.Embedding(num_buckets, num_heads)

    def relative_position_bucket(self, relative_position):
        num_buckets = self.num_buckets
        buckets = torch.zeros_like(relative_position)

        if self.bidirectional:
            num_buckets //= 2
            buckets = buckets + (relative_position > 0).to(torch.long) * num_buckets
            relative_position = torch.abs(relative_position)
        else:
            relative_position = -torch.min(relative_position,
                                           torch.zeros_like(relative_position))

        # half the buckets are exact; the rest widen logarithmically
        max_exact = num_buckets // 2
        is_small = relative_position < max_exact

        large = max_exact + (
            torch.log(relative_position.float().clamp(min=1) / max_exact)
            / math.log(self.max_distance / max_exact)
            * (num_buckets - max_exact)
        ).to(torch.long)
        large = torch.min(large, torch.full_like(large, num_buckets - 1))

        return buckets + torch.where(is_small, relative_position, large)

    def forward(self, seq_len_q, seq_len_k):
        device = self.embeddings_table.weight.device
        range_vec_q = torch.arange(seq_len_q, device=device)
        range_vec_k = torch.arange(seq_len_k, device=device)
        relative_position = range_vec_k[None, :] - range_vec_q[:, None]
        buckets = self.relative_position_bucket(relative_position)

        # (seq_len_q, seq_len_k, num_heads)
        return self.embeddings_table(buckets)
