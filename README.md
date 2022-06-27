# Overview
- After the emergence of Attention, the language models leveraging the attention layer show the best performance in various NLP tasks. Attention allows attending to utilize the most relevant parts of the input sequence by leveraging the attention score which is a weighted result of all of the encoded input vectors simultaneously. Therefore, attention layers are able to increase the learning speed through parallelization without the restrictions appearing in such sequential architectures. This project aims to implement the Transformer Encoder blocks using Absolute Positional Encoding, Relative Position representation of Shaw et al. (2018) and Relative Position representation of Raffel et al. (2019).

- Attention score using Absolute Positional Encoding:
$$\alpha_{ij}^{Abs} = \frac{1}{\sqrt{d}}((w_i+p_i)W^{Q,1})(w_j+p_j)W^{K,1})^T$$
where $w_i$ is word embedding, $p_i$ is absolute positional encoding, $W^{Q,1}$ and $W^{K,1}$ is corresponding weight of query and key.
The absolute positional encoding method is applied to represent the position representation of tokens in Transformer-based architecture. This representation is summed to the word embeddings at the input level.

- Attention score using Relative Position representation of Shaw et al. (2018):
$$\alpha_{ij}^{Rel} = \frac{1}{\sqrt{d}}((w_i+p_i)W^{Q,l})((w_j+p_j)W^{K,l}+a_{j-i}^l)^T$$
where $a_{j-i}^l$ is a learnable parameter that represents the embedding of the relative position $j−i$ in layer $l$.
In order to represent longer sentences in the generalized auto-regressive pretraining model (XLNet), the relative positional encoding is applied to represent the positional representation in multiple segments. This representation is applied at the self-attention-mechanism level, not at the input level.

- Attention score using Relative Position representation of Raffel et al. (2019):
$$\alpha_{ij}^{T5} = \frac{1}{\sqrt{d}}((w_i+p_i)W^{Q,l})((w_j+p_j)W^{K,l})^T+b_{j-i}$$
where $b_{j-i}$ is a learnable parameter that represents the embedding of the relative position $j−i$ and this is shared in all layers.


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


# Prerequisites
- argparse
- torch
- stanza
- spacy
- nltk
- gensim

# Parameters
- nlp_pipeline(str, defaults to "stanza"): NLP preprocessing pipeline.
- unk_ignore(bool, defaults to True): Ignore unseen word or not.
- num_heads(int, defaults to 8): The number of heads for multi-head attention.
- num_layers(int, defaults to 12): The number of transformer encoder blocks.
- positional_encoding(str, defaults to "abs"): Type of positional encoding. (abs, rel, t5)

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
