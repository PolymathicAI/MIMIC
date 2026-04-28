# MIMIC: A Generative Multimodal Foundation Model for Biomolecules

[![Paper](https://img.shields.io/badge/arXiv-2604.24506-b31b1b.svg)](https://arxiv.org/abs/2604.24506)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Model](https://img.shields.io/badge/Model-Release%20in%20Progress-orange.svg)](#open-source-status)
[![Dataset](https://img.shields.io/badge/LORE-Release%20in%20Progress-orange.svg)](#open-source-status)

MIMIC is a generative multimodal foundation model that jointly models DNA, RNA, proteins, and cellular context in one framework.

Most biological AI systems treat sequence, structure, and function as separate tasks. MIMIC instead learns a shared distribution over molecular states, enabling any-to-any inference and design across modalities.


## Why This Matters

- Biological function emerges from coupled constraints across sequence, structure, regulation, and context.
- Single-modality models miss information that is available in complementary modalities.
- Many high-value problems are inverse problems: generate sequences that satisfy desired structural or regulatory outcomes.

## What MIMIC Does

- **Any-to-any generation:** Condition on any observed subset of modalities and infer the rest.
- **Splicing prediction and design:** Improves splice prediction and enables targeted sequence redesign under fixed constraints.
- **Protein design:** Uses multimodal conditioning (e.g., backbone + surface context) to generate diverse high-confidence binders.
- **RNA structure support:** Predicts probing-like reactivity tracks that improve downstream RNA secondary-structure inference.
- **Transfer learning:** Delivers strong performance across diverse RNA and protein downstream benchmarks.

## Architecture at a Glance

- ~1B parameter encoder-decoder transformer
- Split-track multimodal representation (nucleic acid, protein, semantic context, etc.)
- Localized positional encoding within each track
- Register-token compression for global molecular context
- Multi-pathway training for partially observed modality combinations
- Curriculum scaling of context length (1k to 10k tokens)

## LORE Dataset (Training Backbone)

LORE aligns heterogeneous molecular data into coherent, partially observed examples with shared transcript/protein anchors.

Scale highlights:
- 13M RNA transcripts
- 15.5M proteins
- 4B+ natural language tokens
- 6000+ organisms

## Links

- **Paper:** [arXiv:2604.24506](https://arxiv.org/abs/2604.24506)

## Open Source Status

MIMIC model code/weights and LORE release assets are in preparation for public release.

## Citation

If you use this work, please cite:

```bibtex
@article{golkar2026mimic,
  title={MIMIC: A Generative Multimodal Foundation Model for Biomolecules},
  author={Golkar, Siavash et al.},
  year={2026},
  eprint={2604.24506},
  archivePrefix={arXiv},
  primaryClass={q-bio}
}
```
