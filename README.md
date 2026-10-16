# Overview
- After the emergence of Attention, the language models leveraging the attention layer show the best performance in various NLP tasks. Attention allows attending to utilize the most relevant parts of the input sequence by leveraging the attention score which is a weighted result of all of the encoded input vectors simultaneously. Therefore, attention layers are able to increase the learning speed through parallelization without the restrictions appearing in such sequential architectures. This project aims to implement the Transformer Encoder blocks using Absolute Positional Encoding, Relative Position representation of Shaw et al. (2018) and Relative Position representation of Raffel et al. (2019).

- Attention score using Absolute Positional Encoding:
$$\alpha_{ij}^{Abs} = \frac{1}{\sqrt{d}}((w_i+p_i)W^{Q,1})(w_j+p_j)W^{K,1})^T$$
where $w_i$ is word embedding, $p_i$ is absolute positional encoding, $W^{Q,1}$ and $W^{K,1}$ is corresponding weight of query and key.
The absolute positional encoding method is applied to represent the position representation of tokens in Transformer-based architecture. This representation is summed to the word embeddings at the input level.

- Attention score using Relative Position representation of Shaw et al. (2018):
$$\alpha_{ij}^{Rel} = \frac{1}{\sqrt{d}}(w_iW^{Q,l})(w_jW^{K,l}+a_{j-i}^l)^T$$
where $a_{j-i}^l$ is a learnable parameter that represents the embedding of the relative position $j−i$ in layer $l$. Note the absence of $p_i$: Shaw et al. replace the sinusoidal encoding rather than adding to it, so selecting `rel` here does not also apply absolute encoding.
In order to represent longer sentences in the generalized auto-regressive pretraining model (XLNet), the relative positional encoding is applied to represent the positional representation in multiple segments. This representation is applied at the self-attention-mechanism level, not at the input level.

- Attention score using Relative Position representation of Raffel et al. (2019):
$$\alpha_{ij}^{T5} = \frac{1}{\sqrt{d}}(w_iW^{Q,l})(w_jW^{K,l})^T+b_{j-i}$$
where $b_{j-i}$ is a learnable scalar per head, shared across all layers, and again without $p_i$. T5 does not learn one value per distance. It sorts distances into a small number of buckets, exact for nearby positions and widening logarithmically for far ones (32 buckets out to a distance of 128 by default), which is what keeps the table small and lets the bias generalise past the lengths it saw in training.


# Brief description
- text_processing.py
> Output format
> - output: Tokenized result of a given text. (list)
- my_onehot.py
> Output format
> - output: List of tensor of input tokens. (Tensor)
- attentions.py
> Output format
> - output: List of tensor of attention results. (Tensor)
- transformers.py
> Output format
> - output: model (Transformer Encoder Model), output (Last hidden states of the model (Tensor)), output_list (Hidden states of layer 1 to 12 (Tensor)), attn_score_list (Attention scores of layer 1 to 12 (Tensor))
- test_correctness.py
> Properties that have to hold for all three encodings: batch independence, each layer running once, masking taking effect, buffers surviving `.to(device)` and `state_dict`, the T5 bias being reachable from `model.parameters()`, and the sinusoidal formula matching the paper. Run with `python test_correctness.py` or `python -m pytest test_correctness.py`.


# Prerequisites
- argparse
- torch
- stanza
- spacy
- nltk
- gensim

# Parameters
- nlp_pipeline(str, defaults to "spacy"): NLP preprocessing pipeline.
- unk_ignore(bool, defaults to True): Ignore unseen word or not.
- num_heads(int, defaults to 8): The number of heads for multi-head attention. If it does not divide the embedding dimension, the nearest divisor is used and a warning is raised.
- num_layers(int, defaults to 12): The number of transformer encoder blocks.
- positional_encoding(str, defaults to "t5"): Type of positional encoding. (abs, rel, t5)

# References
- Attention: Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., ... & Polosukhin, I. (2017). Attention is all you need. Advances in neural information processing systems, 30.
- Relative Postion Representation: Shaw, P., Uszkoreit, J., & Vaswani, A. (2018). Self-attention with relative position representations. arXiv preprint arXiv:1803.02155.
- Transformer-XL: Dai, Z., Yang, Z., Yang, Y., Carbonell, J., Le, Q. V., & Salakhutdinov, R. (2019). Transformer-xl: Attentive language models beyond a fixed-length context. arXiv preprint arXiv:1901.02860.
- XLNet: Yang, Z., Dai, Z., Yang, Y., Carbonell, J., Salakhutdinov, R. R., & Le, Q. V. (2019). Xlnet: Generalized autoregressive pretraining for language understanding. Advances in neural information processing systems, 32.
- T5 Relative Postion Representation: Raffel, C., Shazeer, N., Roberts, A., Lee, K., Narang, S., Matena, M., ... & Liu, P. J. (2019). Exploring the limits of transfer learning with a unified text-to-text transformer. arXiv preprint arXiv:1910.10683.
- Stanza: Qi, P., Zhang, Y., Zhang, Y., Bolton, J., & Manning, C. D. (2020). Stanza: A Python natural language processing toolkit for many human languages. arXiv preprint arXiv:2003.07082.
- Spacy: Matthew Honnibal and Ines Montani. 2017. spaCy 2: Natural language understanding with Bloom embeddings, convolutional neural networks and incremental parsing. To appear (2017).
- NLTK: Bird, Steven, Edward Loper and Ewan Klein (2009). Natural Language Processing with Python. O'Reilly Media Inc.
- Gensim: Rehurek, R., & Sojka, P. (2010). Software framework for topic modelling with large corpora. In In Proceedings of the LREC 2010 workshop on new challenges for NLP frameworks.


# Corrections (2026-10-16)

An audit of this repository turned up a bug that produced wrong numbers without
raising anything, plus several smaller ones. If you used this code, the first
item is the one that matters.

- **Multi-head attention scrambled the batch.** Queries, keys and values were
  packed as `(batch * heads, ...)` but the result was unpacked as
  `(heads, batch, ...)`. The two orderings disagree, so attention outputs were
  permuted across batch rows for any `batch_size > 1`. No exception was raised.
  It was correct only at `batch_size == 1` or `num_heads == 1`, which is why it
  survived: the example in `main.py` uses a batch of one.
- **Absolute encoding was applied to every run.** `get_output` added sinusoidal
  encoding before branching on the selected scheme, so `rel` was really
  abs+rel and `t5` was abs+t5. No run isolated a single method, which is the
  comparison this repository exists to support. The formulas above have been
  corrected to match Shaw et al. and Raffel et al., both of which replace the
  sinusoidal encoding rather than adding to it.
- **`T5RelativePositionalEncoder` was not T5's scheme.** It learned one
  embedding per clipped distance, which is the Shaw scheme with a different
  output shape. It now implements the logarithmic bucketing from the paper.
  The table dropped from 262,144 rows to 32.
- **The T5 bias never trained.** It was built outside the model, so it never
  appeared in `model.parameters()`. It is now a submodule of `TransformerEncoder`,
  which is also where sharing it across layers belongs.
- **Nothing moved to GPU.** The sinusoidal table was a plain attribute rather
  than a registered buffer, so `.to(device)` left it behind and `state_dict()`
  was empty.
- **Masking and the T5 path crashed.** The mask was shaped `(batch, heads, Lq, Lk)`
  against a `(batch * heads, Lq, Lk)` tensor, and the T5 bias did not broadcast
  past `batch_size == 1`.
- **`num_layers=1` ran the only block twice**, and `get_candidate_heads` ignored
  the requested head count, silently returning the median divisor. Asking for 8
  heads on 768 dimensions gave you 24. It now picks the nearest divisor and warns.

`test_correctness.py` pins all of these. Against the previous code 10 of its 11
checks fail.
