"""Text (BioBERT WordPiece) tokenizer smoke: the ``[text]`` optional extra.

Needs the ``[text]`` extra (transformers) and downloads the BioBERT tokenizer
(``dmis-lab/biobert-base-cased-v1.2``) on first use, so the module skips cleanly
when transformers is missing or the tokenizer cannot be fetched (offline CI).
"""
import pytest

pytest.importorskip("transformers")

# vocab = 28996 BioBERT wordpieces + 3 dummy specials (pad/mask/unk at the tail)
EXPECTED_VOCAB = 28999


@pytest.fixture(scope="module")
def text_tok():
    from mimic.modality_info import MODALITY_INFO

    tok = MODALITY_INFO["tok_funcprot_caption"]["tokenizer"]
    try:
        tok.tokenize("warm up")  # forces the lazy BioBERT load (download on first use)
    except Exception as e:
        pytest.skip(f"BioBERT tokenizer unavailable ({type(e).__name__}): {e}")
    return tok


def test_lazy_vocab_matches_real(text_tok):
    # the lazy shim advertises the same vocab size as the loaded tokenizer
    assert text_tok.vocab_size == EXPECTED_VOCAB
    assert text_tok.real.vocab_size == EXPECTED_VOCAB


def test_tokenize_detokenize_roundtrip(text_tok):
    s = "This protein is a serine/threonine-protein kinase."
    ids = text_tok.tokenize(s)
    assert isinstance(ids, list) and len(ids) > 0
    assert all(0 <= i < EXPECTED_VOCAB for i in ids)
    out = text_tok.detokenize(ids)
    # WordPiece decode round-trips content (spacing/case may be normalized)
    assert "protein" in out and "kinase" in out


def test_shared_across_text_modalities(text_tok):
    """All text-family modalities reuse one BioBERT tokenizer instance."""
    from mimic.modality_info import MODALITY_INFO

    for mod in ("tok_context", "tok_corpus", "tok_gene_family_txt"):
        assert MODALITY_INFO[mod]["tokenizer"] is text_tok
