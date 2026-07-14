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
| protein structure                               | `prot_struct`                            | biotite `AtomArray` (see below) |

> `prot_struct` is decoded to a backbone structure **by default**
> (`detokenize_structure=True`). Decoding ESM3 VQVAE structure tokens back to 3D
> coordinates lazily loads the ESM3 decoder and downloads its weights on first use:
>
> ```python
> out = model.generate("prot_struct")
> out["prot_struct"]     # a biotite AtomArray (backbone structure)
> ```
>
> Pass `detokenize_structure=False` to skip this — `preds` is then `None`. You can
> always get the raw structure token ids with `return_tokens=True`. (The tokenizer's
> `detokenize(tokens, as_biotite=False)` still returns the compact numpy format for
> round-tripping with `tokenize`.)

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

Everything else is cross-track and gated. The track map and allowlist live in the
"Pathway gating" block of
[`src/mimic/modality_info.py`](../src/mimic/modality_info.py):

- `MODALITY_TRACK` / `_SINGLETON_TRACK` — the `protein` / `nucleic` / `text` track of
  each modality.
- `TEXT_ASSOCIATIONS` — which modalities each text channel was trained with.
- `PATHWAY_ALLOWLIST` — explicit cross-track `(input, target)` pathways to enable
  (empty by default); add pairs here to selectively enable cross-track generation.
- `DEFAULT_ON_UNSUPPORTED` — package-wide default (`"error"`); set `"warn"`/`"allow"`
  for a looser install.

## Modalities

MIMIC represents each molecule as a set of co-observed modalities grouped into
three tracks: **nucleic** (RNA/DNA and its per-position annotations), **protein**
(amino-acid sequence, structure, and derived features), and **text** (free-text /
categorical context). Pass any modality to `model.input()` under its short name
(e.g. `rna_seq`) or pre-tokenized under its `tok_` key. The authoritative,
per-checkpoint list is `model.modality_info`. The `Track` column drives the
pathway gating described above. A few assay tracks (`atac`, `cage`, `rasp2`,
`prot_abund`) are **context-conditional**: pass a free-text `context` alongside
them to condition on cell-state / assay metadata — the `Conditioning context`
column shows a real example for each (see `TEXT_ASSOCIATIONS` in
[`src/mimic/modality_info.py`](../src/mimic/modality_info.py)).

| Modality | Track | Description | Example | Conditioning context (`tok_context`) |
|---|---|---|---|---|
| `rna_seq` | nucleic | RNA/DNA nucleotide sequence (unspliced) — the core nucleic input | `"UUUGGAAACUUU…"` | — |
| `cds_junctions` | nucleic | Coding-sequence (CDS) exon–exon junction positions, per position | `"…0001000…"` | — |
| `splice_regions` | nucleic | Splice-region (exon) annotation, per position | `"…0011100…"` | — |
| `splice_jctns_5cls` | nucleic | Per-position 5-class splice-site type: `0`=none, `1`=acceptor, `2`=donor, `3`=TSS (first-exon start), `4`=TES (last-exon end) | `"…00020…0100…"` | — |
| `is_coding` | nucleic | Coding vs. non-coding flag | `[1]` | — |
| `feature_type` | nucleic | Genomic feature-type label | `['protein_coding']` | — |
| `phylop_human` | nucleic | phyloP evolutionary-conservation score (human), per position | `[-0.66, 1.04, …]` | — |
| `phylop_mouse` | nucleic | phyloP evolutionary-conservation score (mouse), per position | `[-0.26, -0.92, …]` | — |
| `atac` | nucleic | ATAC-seq chromatin-accessibility signal, per position (`N` = unmeasured). **Cell-state-conditional** | `"…NNNN…"` | `"human, GM23338 lymphoblastoid cell line (EBV-transformed B lymphocyte)"` |
| `cage` | nucleic | CAGE transcription-start signal, per position. **Cell-state-conditional** | `[0.001, 0.001, …]` | `"skeletal muscle, human, fetal"` |
| `rasp2` | nucleic | RASP2 (icSHAPE-style) RNA-structure reactivity, per position (`nan` where unmeasured). **Condition-conditional** | `[nan, 0.42, …]` | `"technology: icSHAPE, reagent: NAI-N3, in vivo, cell line: K562, human"` |
| `aa_seq` | protein | Amino-acid (protein) sequence — the core protein input | `"MTPPERLFLP…"` | — |
| `rna_codons` | protein | Codon sequence aligned to the protein (nucleotide content, protein-aligned track) | `['AUG', 'ACA', 'CCA', …]` | — |
| `prot_struct` | protein | Protein 3D structure as ESM3 VQVAE tokens (decode to a backbone via `detokenize_structure`) | `[754, 1510, 49, …]` | — |
| `dssp` | protein | DSSP secondary-structure class, per residue | `"CCXX…HHH…"` | — |
| `sasa` | protein | Solvent-accessible surface area, per residue | `[225.1, 128.6, …]` | — |
| `prot_abund` | protein | Protein abundance (PaxDb ppm), scalar. **Cell-state-conditional** | `[385.6]` | `"Leptospira interrogans (bacterium), control"` |
| `funcprot_caption` | protein | Free-text protein functional caption | `"Catalyzes the hydrolysis of…"` | — |
| `masif_charge` | protein | MaSIF surface Poisson–Boltzmann charge, per vertex | `[6.9, -3.3, …]` | — |
| `masif_hbond` | protein | MaSIF surface hydrogen-bond potential, per vertex | `[-1.77, -1.64, …]` | — |
| `masif_hydrophobicity` | protein | MaSIF surface hydrophobicity, per vertex | `[0.32, -0.31, …]` | — |
| `masif_si_index` | protein | MaSIF surface shape-index, per vertex | `[0.34, 0.22, …]` | — |
| `masif_n_vertices` | protein | MaSIF surface vertex count, per patch | `[65.0, 42.0, …]` | — |
| `context` | text | Free-text semantic context (e.g. cell-state / assay) — the conditioning channel itself | `"HepG2 cell line"` | — |
| `corpus` | text | Free-text corpus / source label | `"UniProtKB/Swiss-Prot"` | — |
| `gene_family_txt` | text | Free-text gene-family description | `"Zinc-finger C2H2 family"` | — |
