# `generate()` — output format and pathway safety

This note documents two aspects of `model.generate(...)`: what it returns, and how
it guards against untrained conditioning→target combinations.

## Return format

`generate` returns a dict keyed by the requested target name(s). **By default, each
value is the detokenized generation for that modality** — not a bundle of tensors.

```python
model.input([{"rna_seq": "ACGUACGUACGUACGUACGUACGU"}])

out = model.generate("sasa")
out["sasa"]          # -> np.ndarray of SASA values (bin centers)

out = model.generate("aa_seq")
out["aa_seq"]        # -> "MKT..." (a string)
```

The type of `preds` follows the modality's tokenizer:

| modality kind                                   | example targets                          | `preds` type            |
| ----------------------------------------------- | ---------------------------------------- | ----------------------- |
| character / sequence / text                     | `aa_seq`, `rna_seq`, `dssp`, `splice_jctns_5cls`, `cds`, `context` | `str` (one char per position) |
| scalar track (digitized)                        | `sasa`, `phylop_human`, `cage`, `atac`   | `np.ndarray` of floats  |
| boolean track                                   | `is_coding`                              | list of `bool`          |
| categorical track                               | `kingdom`, `feature_type`                | list of class labels    |
| protein structure                               | `prot_struct`                            | biotite `AtomArray`, or `None` (see below) |

> `prot_struct` returns `preds=None` **by default**: decoding ESM3 VQVAE structure
> tokens back to 3D coordinates is expensive (it lazily loads the ESM3 decoder and
> downloads its weights on first use) and needs the `[structure]` extra. It is opt-in:
>
> ```python
> out = model.generate("prot_struct", detokenize_structure=True)
> out["prot_struct"]     # a biotite AtomArray (backbone structure)
> ```
>
> You can always get the raw structure token ids instead with `return_tokens=True`.
> (The tokenizer's `detokenize(tokens, as_biotite=False)` still returns the compact
> numpy format for round-tripping with `tokenize`.)

### Getting the raw model outputs

Set any of the return flags to also surface the underlying strategy outputs. When
**any** flag is `True`, each target's value becomes a dict with `"preds"` plus only
the extras you asked for (as numpy arrays):

```python
out = model.generate(
    "sasa",
    return_logits=True,
    return_probs=True,
)
out["sasa"]["preds"]     # detokenized SASA array (always present)
out["sasa"]["logits"]    # (L, vocab) logits
out["sasa"]["probs"]     # (L, vocab) model marginal probabilities
# "tokens" and "sampling_probs" are absent — they weren't requested
```

| flag                     | key in the returned dict | meaning                                              |
| ------------------------ | ------------------------ | ---------------------------------------------------- |
| `return_tokens`          | `tokens`                 | generated token ids (integer array)                  |
| `return_logits`          | `logits`                 | raw logits over the vocabulary                       |
| `return_probs`           | `probs`                  | model marginal probabilities (softmax of logits)     |
| `return_sampling_probs`  | `sampling_probs`         | post-temperature probabilities used to sample tokens |

All four default to `False`. With no flag set, the return is the bare `preds`.

## Pathway gating

MIMIC is only trained on certain `(conditioning → target)` combinations. Generating a
target MIMIC was **not** trained to produce from the given inputs — e.g. conditioning
on a protein sequence and generating the DNA/RNA `is_coding` track — is a cross-track
pathway whose output is not meaningful, so `generate` **raises by default**:

```python
model.input([{"aa_seq": "MKTAYIAKQR"}])
model.generate("is_coding")
# ValueError: Untrusted generation pathway -- MIMIC was not trained to generate these
#   targets from the given conditioning (is_coding: nucleic-track target not reachable
#   from inputs (tracks: ['protein'])). To run it anyway pass on_unsupported='allow'
#   (or 'warn' to run with a warning).
```

Override per call with `on_unsupported`:

```python
model.generate("is_coding", on_unsupported="error")  # default: raise, does not run
model.generate("is_coding", on_unsupported="warn")   # run, but log a loud warning
model.generate("is_coding", on_unsupported="allow")  # run silently, no check
```

### What's allowed

A requested target is supported when some conditioning input reaches it via one of:

- **within-track** — input and target are in the same molecular track (`protein` or
  `nucleic`); e.g. `aa_seq → sasa`, `rna_seq → splice_jctns_5cls`, `dssp → aa_seq`.
- **text association** — a text channel paired with a modality it was trained with
  (`context ↔ {atac, cage, rasp2, prot_abund}`).
- **RNA ↔ protein sequence** — `aa_seq ↔ rna_seq` translation (an explicit allowlist entry).
  Because the two are in different summation groups, pass `target_lens=<int>` to set the
  generated length (it can't be inferred from a different-group input).

Everything else is cross-track and gated. The track map and allowlist live in the
"Pathway gating" block of
[`src/mimic/modality_info.py`](../src/mimic/modality_info.py):

- `MODALITY_TRACK` / `_SINGLETON_TRACK` — the `protein` / `nucleic` / `text` track of
  each modality.
- `TEXT_ASSOCIATIONS` — which modalities each text channel was trained with.
- `PATHWAY_ALLOWLIST` — explicit cross-track `(input, target)` pathways to enable
  (ships with `aa_seq ↔ rna_seq`); add pairs here to selectively enable more.
- `DEFAULT_ON_UNSUPPORTED` — package-wide default (`"error"`); set `"warn"`/`"allow"`
  for a looser install.
