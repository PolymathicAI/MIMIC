# LORE — third-party data sources and attribution

LORE is a derived work. Every modality in it was computed from a public upstream
resource, and several of those resources are distributed under licences that require
attribution when redistributed. This file records, per modality, what the source was
and under what terms it is being passed on.

The LORE **tooling and pipeline code** in this directory are MIT-licensed (see the
repository `LICENSE`). This file concerns the **data**.

> **Status: two entries are unresolved and are marked ⚠ below. They must be settled
> before the dataset is made public.** They are flagged rather than guessed at.

## Per-source record

| Upstream source | Modalities derived from it | Licence / terms | Redistribution |
|:-|:-|:-|:-|
| [AlphaFold Protein Structure Database](https://alphafold.ebi.ac.uk/) v4, pLDDT ≥ 70 | `aa_seq`, `prot_struct`, `dssp`, `sasa`, `masif_*` | [CC-BY-4.0](https://alphafold.ebi.ac.uk/assets/License-Disclaimer.pdf) | Permitted with attribution |
| [UniProt](https://www.uniprot.org/) / Swiss-Prot | `funcprot_caption` | CC-BY-4.0 | Permitted with attribution |
| [NCBI RefSeq](https://www.ncbi.nlm.nih.gov/refseq/) | `rna_seq`, `splice_jctns_5cls`, `splice_regions`, `cds_junctions`, `rna_codons`, `is_coding`, `feature_type`, `gene_family_txt` | US Government work, public domain | Unrestricted |
| [UCSC Genome Browser](https://genome.ucsc.edu/) phyloP (hg38 100-way, mm39 35-way) | `phylop_human`, `phylop_mouse` | [Freely usable for any purpose](https://genome.ucsc.edu/conditions.html), including commercial (only liftOver chain files are non-commercial, and are not used here) | Unrestricted |
| [ENCODE](https://www.encodeproject.org/) ATAC-seq narrowPeak | `atac` | Released without restriction; citation expected | Permitted |
| [PaxDb](https://pax-db.org/) v5.0 | `prot_abund` | CC-BY-4.0 | Permitted with attribution |
| [FANTOM5](https://fantom.gsc.riken.jp/5/) phase 2.6 CAGE peaks | `cage` | ⚠ **Unresolved.** No data licence is stated in the RIKEN `datafiles/` tree; the accompanying publications are CC-BY-4.0, which does not by itself license the data. | ⚠ Confirm with RIKEN |
| [RASP v2.0](https://academic.oup.com/nar/article/53/D1/D211/7901281) RNA structure probing | `rasp2` | ⚠ **Unresolved.** The frequently-cited "CC BY-NC" is Oxford University Press's *article* licence, not a licence over the database contents. The project host (`rasp.zhanglab.net`) no longer serves a valid certificate. | ⚠ Contact the RASP authors |
| [common-pile/pubmed_filtered](https://huggingface.co/datasets/common-pile/pubmed_filtered) | `corpus` | Mixture of **CC0, CC-BY and CC-BY-SA** (drawn from the PMC Open Access Subset) | ⚠ The CC-BY-SA component imposes share-alike on derived text; see note below |
| [ESM3](https://github.com/evolutionaryscale/esm) `esm3-sm-open-v1` structure VQVAE | tokenizer for `prot_struct` | **MIT**, © 2026 Chan Zuckerberg Biohub, Inc. | Permitted |
| [BioBERT](https://huggingface.co/dmis-lab/biobert-base-cased-v1.2) | tokenizer for `corpus`, `context`, `funcprot_caption`, `gene_family_txt` | Apache-2.0 | Permitted |

### Note on `prot_struct` and ESM3

`prot_struct` values are codebook indices produced by the ESM3 structure VQVAE. ESM3-open
was previously distributed under the EvolutionaryScale Cambrian **Non-Commercial** License
Agreement, which constrained outputs and derivative works. Following the Chan Zuckerberg
Biohub acquisition it is now MIT-licensed, and the Hugging Face repository is no longer
gated. Redistributing these tokens is therefore unencumbered. The mirror at
`biohub/esm3-sm-open-v1` carries the same terms.

### Note on `corpus`

`corpus` is BioBERT token ids over openly-licensed PubMed Central full text. Common Pile
selects only articles whose journal declared CC0, CC-BY or CC-BY-SA, so nothing here is
non-commercial or no-derivatives — but the CC-BY-SA fraction means the derived text is
arguably subject to share-alike. `corpus` is 501,381 rows / 2.55 GB of general biomedical
prose and carries no central-dogma content, so the cheapest resolutions are to omit it or
to license that configuration separately. **Decide before publishing.**

## Citing the upstream sources

If you use LORE, please cite MIMIC and the upstream resources for whichever modalities you
actually use. At minimum, work touching protein structure should cite AlphaFold DB and
UniProt; work touching transcripts should cite RefSeq; work touching conservation should
cite UCSC/phyloP.

## Things deliberately absent

The build pipeline contains download code for **GTEx**, **ARCHS4** and **Cistrome DB**.
None of the three contributes any modality to this release, and no data derived from them
is redistributed here. GTEx in particular carries access conditions that would not permit
this kind of republication; it was not used.
