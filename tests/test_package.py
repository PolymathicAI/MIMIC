"""End-to-end package smoke: load -> input -> embed -> masked generate.

Requires released weights (config.json + model.safetensors). Point `MIMIC_RELEASE_DIR`
at a local release dir, or the test is skipped (so a fresh clone without weights still
passes). Runs on CPU.
"""
import os

import pytest

RELEASE_DIR = os.environ.get("MIMIC_RELEASE_DIR")

pytestmark = pytest.mark.skipif(
    not (RELEASE_DIR and os.path.exists(os.path.join(RELEASE_DIR, "model.safetensors"))),
    reason="set MIMIC_RELEASE_DIR to a local release dir (config.json + model.safetensors)",
)


@pytest.fixture(scope="module")
def model():
    from mimic import load_pretrained
    return load_pretrained(local_path=RELEASE_DIR, device="cpu")


def test_embed(model):
    model.input([{"rna_seq": "ACGUACGUACGUACGU"}])
    reps = model.embed(sep_encodings=False)
    assert reps["full"].isfinite().all()
    assert reps["full"].shape[-1] == 1536


def test_masked_generation(model):
    rna = model.tokenizers["tok_rna_seq"]
    ids = list(rna.tokenize("ACGUACGUACGUACGUACGUACGU"))
    ids[8:16] = [rna.mask_token_id] * 8
    model.input([{"tok_rna_seq": ids}])
    out = model.generate("rna_seq", verbose=False)
    preds = out["rna_seq"]                  # default return: detokenized string directly
    assert isinstance(preds, str)
    assert len(preds) == 8                  # one nucleotide per masked position


def test_cross_modal_generation(model):
    model.input([{"rna_seq": "ACGUACGUACGUACGUACGUACGU"}])
    out = model.generate("splice_jctns_5cls", verbose=False)
    assert out["splice_jctns_5cls"]         # default return is the detokenized preds


def test_generate_return_flags(model):
    """Rich return: any return_* flag makes each value a dict with preds + extras."""
    model.input([{"rna_seq": "ACGUACGUACGUACGUACGUACGU"}])
    out = model.generate("splice_jctns_5cls", verbose=False,
                         return_logits=True, return_probs=True)
    entry = out["splice_jctns_5cls"]
    assert set(entry) >= {"preds", "logits", "probs"}
    assert "tokens" not in entry and "sampling_probs" not in entry  # only requested extras


def test_pathway_gate_cross_track_raises_by_default(model):
    """Cross-track pathway (protein input -> nucleic is_coding) raises by default; allow runs it."""
    model.input([{"aa_seq": "MKTAYIAKQR"}])
    with pytest.raises(ValueError, match="Untrusted generation pathway"):
        model.generate("is_coding", verbose=False)                 # default = error
    out = model.generate("is_coding", verbose=False, on_unsupported="allow")
    assert "is_coding" in out


def test_pathway_gate_allows_within_track(model):
    """Within-track pathway (RNA input -> nucleic is_coding) runs even in error mode."""
    model.input([{"rna_seq": "ACGUACGUACGUACGUACGUACGU"}])
    out = model.generate("is_coding", verbose=False, on_unsupported="error")
    assert "is_coding" in out


def test_pathway_gate_allows_seq_translation(model):
    """RNA<->protein sequence translation is allowlisted, so not gated (default error mode).

    aa_seq is a different summation group than rna_seq, so its length can't be inferred
    and target_lens must be given -- the point here is that the gate does NOT raise.
    """
    model.input([{"rna_seq": "ACGUACGUACGUACGUACGUACGU"}])
    out = model.generate("aa_seq", verbose=False, target_lens=8)   # allowlisted; explicit length
    assert "aa_seq" in out and isinstance(out["aa_seq"], str)
