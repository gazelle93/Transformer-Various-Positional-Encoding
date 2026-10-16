"""Properties that must hold for any of the three positional encodings.

Run: python -m pytest test_correctness.py   (or just: python test_correctness.py)

The batch-independence test exists because the head/batch reshape round-trip
disagreed with itself: queries were packed (batch * head) and unpacked
(head, batch), which silently permuted outputs across the batch for every
batch_size > 1. It raised no error, so only a property test catches it.
"""
import torch

from attentions import MultiHeadAttention
from positional_encoders import (AbsolutePositionalEncoder,
                                 RelativePositionalEncoder,
                                 T5RelativePositionalEncoder)
from transformers import TransformerEncoder, get_candidate_heads

PE_TYPES = ["abs", "rel", "t5"]


def _encoder(pe, emb_dim=8, num_heads=4):
    torch.manual_seed(0)
    return TransformerEncoder(emb_dim, num_heads, num_layers=2,
                              positional_encoding=pe).eval()


def test_batch_independence():
    """Row i of a batched forward must equal that row run on its own."""
    for pe in PE_TYPES:
        enc = _encoder(pe)
        x = torch.randn(4, 6, 8)
        with torch.no_grad():
            batched = enc(x)[0]
            for i in range(x.size(0)):
                solo = enc(x[i:i + 1])[0]
                assert torch.allclose(batched[i], solo[0], atol=1e-5), \
                    f"{pe}: row {i} differs when run in a batch"


def test_num_layers_are_each_used_once():
    for n in (1, 2, 3):
        enc = TransformerEncoder(8, 4, num_layers=n).eval()
        seen = []
        for i, blk in enumerate(enc.encoder_blocks):
            fwd = blk.forward
            blk.forward = (lambda i, fwd: (lambda *a, **k: (seen.append(i), fwd(*a, **k))[1]))(i, fwd)
        with torch.no_grad():
            enc(torch.randn(2, 5, 8))
        assert seen == list(range(n)), f"num_layers={n}: blocks ran {seen}"


def test_mask_is_applied():
    for pe in PE_TYPES:
        enc = _encoder(pe)
        x = torch.randn(2, 5, 8)
        mask = torch.zeros(2, 5, 5, dtype=torch.bool)
        mask[:, :, 3:] = True                      # forbid the last two keys
        with torch.no_grad():
            _, _, attn = enc(x, mask=mask)
        assert attn[0][..., 3:].max() < 1e-6, f"{pe}: masked keys got attention"


def test_absolute_encoding_is_only_applied_when_selected():
    """rel and t5 are defined against the raw embedding, not on top of abs.

    Captures what get_output actually hands the encoder. Adding sinusoidal
    encoding unconditionally makes every "rel" run really abs+rel, so no run
    isolates a single scheme and the comparison the repo exists for is void.
    """
    import transformers as T
    seen = {}
    original = T.TransformerEncoder.forward

    def capture(self, input_embedding, *a, **k):
        seen["x"] = input_embedding.clone()
        return original(self, input_embedding, *a, **k)

    T.TransformerEncoder.forward = capture
    try:
        x = torch.randn(1, 5, 8)
        for pe in ("rel", "t5"):
            T.get_output(x, 2, pe, 4)
            assert torch.allclose(seen["x"], x), \
                f"{pe}: input was modified, absolute encoding leaked in"
        T.get_output(x, 2, "abs", 4)
        assert not torch.allclose(seen["x"], x), \
            "abs: absolute encoding was not applied"
    finally:
        T.TransformerEncoder.forward = original


def test_modules_move_with_to_and_round_trip_state_dict():
    for pe in PE_TYPES:
        enc = _encoder(pe)
        sd = enc.state_dict()
        clone = _encoder(pe)
        clone.load_state_dict(sd)                  # raises on missing buffers
        x = torch.randn(2, 5, 8)
        with torch.no_grad():
            assert torch.allclose(enc(x)[0], clone(x)[0], atol=1e-6)


def test_t5_bias_is_trainable_through_the_model():
    enc = _encoder("t5")
    names = [n for n, _ in enc.named_parameters()]
    assert any("relative" in n for n in names), \
        "the T5 bias table is not registered on the model, so it cannot train"


def test_absolute_encoding_matches_the_paper():
    import math
    emb_dim, pos = 8, 3
    pe = AbsolutePositionalEncoder(emb_dim)(torch.zeros(1, 5, emb_dim))[0, pos]
    for i in range(emb_dim // 2):
        denom = 10000 ** (2 * i / emb_dim)
        assert math.isclose(pe[2 * i].item(), math.sin(pos / denom), abs_tol=1e-5)
        assert math.isclose(pe[2 * i + 1].item(), math.cos(pos / denom), abs_tol=1e-5)


def test_odd_embedding_dimension_is_rejected_clearly():
    try:
        AbsolutePositionalEncoder(7)
    except ValueError:
        return
    raise AssertionError("odd emb_dim should raise ValueError, not crash later")


def test_candidate_heads_picks_the_nearest_divisor():
    import warnings
    assert get_candidate_heads(768, 8) == 8          # already valid, no warning
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        assert get_candidate_heads(768, 7) == 8      # 6 and 8 tie; larger wins
        assert get_candidate_heads(100, 8) == 10     # 10 is nearer than 5
        assert len(w) == 2, "a silent head change is the bug this guards"
    assert 100 % get_candidate_heads(100, 8) == 0


def test_t5_buckets_are_logarithmic_and_bounded():
    enc = T5RelativePositionalEncoder(num_heads=4, num_buckets=32)
    b = enc.relative_position_bucket(torch.arange(-300, 300)[None, :])
    assert int(b.max()) < 32 and int(b.min()) >= 0
    near = enc.relative_position_bucket(torch.tensor([[1, 2]]))
    assert near[0, 0] != near[0, 1], "adjacent near positions should not share a bucket"
    far = enc.relative_position_bucket(torch.tensor([[200, 201]]))
    assert far[0, 0] == far[0, 1], "distant positions should share a bucket"


def test_relative_table_is_not_oversized():
    t5 = T5RelativePositionalEncoder(num_heads=4)
    assert t5.embeddings_table.num_embeddings <= 64, \
        "bucketed bias should need tens of rows, not max_position squared"


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = []
    for fn in fns:
        try:
            fn()
            print(f"  ok    {fn.__name__}")
        except Exception as e:
            failed.append(fn.__name__)
            msg = (str(e).splitlines() or [type(e).__name__])[0]
            print(f"  FAIL  {fn.__name__}: {msg[:88]}")
    print(f"\n{len(fns)-len(failed)} passed, {len(failed)} failed")
    raise SystemExit(1 if failed else 0)
