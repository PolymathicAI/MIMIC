# Modality registry for the public MIMIC package.
# Light tokenizers (pure-python / npy-npz bins) are imported eagerly. Text
# (BioBERT) and structure (ESM3) tokenizers are wrapped lazily via _LazyTokenizer
# so building the model does not import `transformers` / `biotite` or download
# ESM3 weights; those load on first tokenize/detokenize.

import hashlib
from functools import partial

from .model.encoder_embeddings import (
    ChainTokenEncoderEmbedding,
    SequenceEncoderEmbedding,
)
from .model.decoder_embeddings import (
    ChainTokenDecoderEmbedding,
    SequenceDecoderEmbedding,
)

# --- light tokenizers (no heavy dependencies) ---
from .tokenizers.dna.phylop.phylop_tokenizer import (
    phylop_mouse_tokenizer,
    phylop_human_tokenizer,
)
from .tokenizers.dna.atac.atac_tokenizer import atac_tokenizer
from .tokenizers.rna.rna_seq.rna_seq_tokenizer import rna_seq_tokenizer
from .tokenizers.rna.splice_regions.splice_regions_tokenizer import splice_regions_tokenizer
from .tokenizers.rna.splice_junctions.splice_junctions_tokenizer import splice_junctions_tokenizer
from .tokenizers.rna.splice_jctns_5cls.splice_jctns_5cls_tokenizer import splice_jctns_5cls_tokenizer
from .tokenizers.rna.cds.cds_tokenizer import cds_tokenizer
from .tokenizers.rna.utr.utr_tokenizer import utr_tokenizer
from .tokenizers.rna.cds_junctions.cds_junctions_tokenizer import cds_junctions_tokenizer
from .tokenizers.rna.kingdom.kingdom_tokenizer import kingdom_tokenizer
from .tokenizers.rna.is_coding.is_coding_tokenizer import is_coding_tokenizer
from .tokenizers.rna.feature_type.feature_type_tokenizer import feature_type_tokenizer
from .tokenizers.protein.aa_seq.aa_seq_tokenizer import aa_seq_tokenizer
from .tokenizers.protein.dssp.dssp_tokenizer import dssp_tokenizer
from .tokenizers.protein.sasa.sasa_tokenizer import sasa_tokenizer
from .tokenizers.protein.abundance.prot_abundance_tokenizer import prot_abund_tokenizer
from .tokenizers.protein.masif.masif_tokenizer import (
    n_vertices_tokenizer, hbond_tokenizer, hydrophobicity_tokenizer,
    charge_tokenizer, si_index_tokenizer,
)
from .tokenizers.rna.rna_codons.rna_codons_tokenizer import rna_codons_tokenizer
from .tokenizers.rna.rasp2.rasp2_tokenizer import rasp2_tokenizer
from .tokenizers.rna.cage.cage_tokenizer import cage_tokenizer


def generate_uint15_hash(seed_str: str) -> int:
    """Deterministic string -> unsigned int15 id (stable modality/group id)."""
    return int(hashlib.sha256(seed_str.encode("utf-8")).hexdigest(), 16) % (2**15)


class _LazyTokenizer:
    """Placeholder tokenizer for heavy modalities (text, structure).

    Exposes vocab size and special-token ids as constants so the model can be
    constructed without importing the heavy dependency (transformers / biotite /
    ESM3). The real tokenizer is imported/constructed on first tokenize/detokenize
    (or other attribute access).
    """

    def __init__(self, loader, *, vocab_size, pad_token_id, mask_token_id,
                 unk_token_id, has_weighted_mean=False):
        self._loader = loader
        self._real = None
        self.vocab_size = vocab_size
        self.pad_token_id = pad_token_id
        self.mask_token_id = mask_token_id
        self.unk_token_id = unk_token_id
        self.has_weighted_mean = has_weighted_mean

    @property
    def real(self):
        if self._real is None:
            self._real = self._loader()
        return self._real

    def tokenize(self, *args, **kwargs):
        return self.real.tokenize(*args, **kwargs)

    def detokenize(self, *args, **kwargs):
        return self.real.detokenize(*args, **kwargs)

    def __getattr__(self, name):
        # Only reached for attributes not set in __init__ (e.g. .init, .token_to_id).
        if name.startswith("_"):
            raise AttributeError(name)
        return getattr(self.real, name)


def _load_text_tokenizer():
    from .tokenizers.protein.function.funcprot_caption_tokenizer import text_tokenizer
    return text_tokenizer


def _load_struct_tokenizer():
    from .tokenizers.protein.structure.struct_tokenizer import esm3_tokenizer
    return esm3_tokenizer


# Text (BioBERT WordPiece): vocab 28999 = 28996 BioBERT + 3 dummy specials.
text_tokenizer = _LazyTokenizer(
    _load_text_tokenizer,
    vocab_size=28999, pad_token_id=28996, mask_token_id=28997, unk_token_id=28998,
    has_weighted_mean=False,
)
# Structure (ESM3 VQVAE): vocab 4104 = 4096 codebook + 8 specials.
esm3_tokenizer = _LazyTokenizer(
    _load_struct_tokenizer,
    vocab_size=4104, pad_token_id=4101, mask_token_id=4102, unk_token_id=4103,
    has_weighted_mean=False,
)

def partial_init_embeddings(modality_info):
    """Goes through the modalities and does a partial
    init of the embedding functions
    """

    for modality, info in modality_info.items():
        tokenizer = info["tokenizer"]
        vs = tokenizer.vocab_size
        max_seq_len = info["max_seq_len"]
        pad_token_id = tokenizer.pad_token_id
        mask_token_id = tokenizer.mask_token_id
        unk_token_id = tokenizer.unk_token_id

        if "class_balance" in info:
            assert len(info["class_balance"]) == vs, \
                f"Modality {modality} has class balance length {len(info['class_balance'])} != vocab size {vs}"

        if modality_info[modality]["type"] == "chain_token":
            encoder_embedding = ChainTokenEncoderEmbedding
            decoder_embedding = ChainTokenDecoderEmbedding
        elif modality_info[modality]["type"] == "text_token_all_targets":
            encoder_embedding = SequenceEncoderEmbedding
            decoder_embedding = SequenceDecoderEmbedding
        else:
            raise ValueError(f"Unknown modality type {modality_info[modality]['type']}")

        # Prepare encoder embedding arguments
        modality_info[modality]["encoder_embedding"] = partial(
            encoder_embedding,
            vocab_size=vs,
            max_length=max_seq_len,
            pad_token_id=pad_token_id,
            mask_token_id=mask_token_id,
        )

        modality_info[modality]["decoder_embedding"] = partial(
            decoder_embedding,
            vocab_size=vs,
            max_length=max_seq_len,
            pad_token_id=pad_token_id,
        )

        modality_info[modality]["vocab_size"] = vs
        modality_info[modality]["pad_token_id"] = pad_token_id
        modality_info[modality]["unk_token_id"] = unk_token_id
        modality_info[modality]["id"] = generate_uint15_hash(modality)
        modality_info[modality]["pretokenized"] = True


# Modalities are:
# group 0: (protein modalities) aa_seq, dssp, sasa, prot_struct
# group 1: (dna/rna modalities) rna_seq, phylop_mouse, phylop_human, exon, cds
# ungrouped: kingdom

# protein
GROUP_0_MIN_TOK = 0
GROUP_0_MAX_TOK = 'context_len'
GROUP_0_MAX_SEQ_LEN = 'context_len'
GROUP_0_NAME = "protein"

# dna
GROUP_1_MIN_TOK = 0
GROUP_1_MAX_TOK = 'context_len'
GROUP_1_MAX_SEQ_LEN = None
GROUP_1_NAME = "dna/rna"

MODALITY_INFO = {
    # GROUP 0: PROTEIN MODALITIES
    "tok_aa_seq": {
        "type": "chain_token",
        "tokenizer": aa_seq_tokenizer,
        "min_tokens": GROUP_0_MIN_TOK,
        "max_tokens": GROUP_0_MAX_TOK,
        "max_seq_len": GROUP_0_MAX_SEQ_LEN,
        "summation_group": 0,
        "target_cont_ratio": 0.05,
    },
    "tok_dssp": {
        "type": "chain_token",
        "tokenizer": dssp_tokenizer,
        "min_tokens": GROUP_0_MIN_TOK,
        "max_tokens": GROUP_0_MAX_TOK,
        "max_seq_len": GROUP_0_MAX_SEQ_LEN,
        "summation_group": 0,
        "target_cont_ratio": 0.05,
    },
    "tok_sasa": {
        "type": "chain_token",
        "tokenizer": sasa_tokenizer,
        "min_tokens": GROUP_0_MIN_TOK,
        "max_tokens": GROUP_0_MAX_TOK,
        "max_seq_len": GROUP_0_MAX_SEQ_LEN,
        "summation_group": 0,
        "target_cont_ratio": 0.05,
    },
    "tok_masif_n_vertices": {
        "type": "chain_token",
        "tokenizer": n_vertices_tokenizer,
        "min_tokens": GROUP_0_MIN_TOK,
        "max_tokens": GROUP_0_MAX_TOK,
        "max_seq_len": GROUP_0_MAX_SEQ_LEN,
        "summation_group": 0,
        "target_cont_ratio": 0.05,
    },
    "tok_masif_hbond": {
        "type": "chain_token",
        "tokenizer": hbond_tokenizer,
        "min_tokens": GROUP_0_MIN_TOK,
        "max_tokens": GROUP_0_MAX_TOK,
        "max_seq_len": GROUP_0_MAX_SEQ_LEN,
        "summation_group": 0,
        "target_cont_ratio": 0.05,
    },
    "tok_masif_hydrophobicity": {
        "type": "chain_token",
        "tokenizer": hydrophobicity_tokenizer,
        "min_tokens": GROUP_0_MIN_TOK,
        "max_tokens": GROUP_0_MAX_TOK,
        "max_seq_len": GROUP_0_MAX_SEQ_LEN,
        "summation_group": 0,
        "target_cont_ratio": 0.05,
    },
    "tok_masif_charge": {
        "type": "chain_token",
        "tokenizer": charge_tokenizer,
        "min_tokens": GROUP_0_MIN_TOK,
        "max_tokens": GROUP_0_MAX_TOK,
        "max_seq_len": GROUP_0_MAX_SEQ_LEN,
        "summation_group": 0,
        "target_cont_ratio": 0.05,
    },
    "tok_masif_si_index": {
        "type": "chain_token",
        "tokenizer": si_index_tokenizer,
        "min_tokens": GROUP_0_MIN_TOK,
        "max_tokens": GROUP_0_MAX_TOK,
        "max_seq_len": GROUP_0_MAX_SEQ_LEN,
        "summation_group": 0,
        "target_cont_ratio": 0.05,
    },
    # Prot struct tokenzier not yet merged
    "tok_prot_struct": {
        "type": "chain_token",
        "tokenizer": esm3_tokenizer,
        "min_tokens": GROUP_0_MIN_TOK,
        "max_tokens": GROUP_0_MAX_TOK,
        "max_seq_len": GROUP_0_MAX_SEQ_LEN,
        "summation_group": 0,
        "target_cont_ratio": 0.05,
    },
    "tok_rna_codons": {
        "type": "chain_token",
        "tokenizer": rna_codons_tokenizer,
        "min_tokens": GROUP_0_MIN_TOK,
        "max_tokens": GROUP_0_MAX_TOK,
        "max_seq_len": GROUP_0_MAX_SEQ_LEN,
        "summation_group": 0,
        "target_cont_ratio": 0.05,
    },
    # GROUP 1: RNA MODALITIES
    "tok_rna_seq": {
        "type": "chain_token",
        "tokenizer": rna_seq_tokenizer,
        "min_tokens": GROUP_1_MIN_TOK,
        "max_tokens": GROUP_1_MAX_TOK,
        "max_seq_len": GROUP_1_MAX_SEQ_LEN,
        "summation_group": 1,
        "target_cont_ratio": 0.05,
    },
    "tok_rasp2": {
        "type": "chain_token",
        "tokenizer": rasp2_tokenizer,
        "min_tokens": GROUP_1_MIN_TOK,
        "max_tokens": GROUP_1_MAX_TOK,
        "max_seq_len": GROUP_1_MAX_SEQ_LEN,
        "summation_group": 1,
        "target_cont_ratio": 0.05,
    },
    "tok_cage": {
        "type": "chain_token",
        "tokenizer": cage_tokenizer,
        "min_tokens": GROUP_1_MIN_TOK,
        "max_tokens": GROUP_1_MAX_TOK,
        "max_seq_len": GROUP_1_MAX_SEQ_LEN,
        "summation_group": 1,
        "target_cont_ratio": 0.05,
        "class_balance": [0.99152, 0.00089, 0.00068, 0.00054, 0.00049, 0.00044, 0.00043, 0.00039, 0.00038, 0.00036, 0.00037, 0.00037, 0.00039, 0.00041, 0.00041, 0.00042, 0.00042, 0.0004, 0.00038, 0.0003, 1.0, 1.0, 1.0],
    },
    "tok_splice_regions": {
        "type": "chain_token",
        "tokenizer": splice_regions_tokenizer,
        "min_tokens": GROUP_1_MIN_TOK,
        "max_tokens": GROUP_1_MAX_TOK,
        "max_seq_len": GROUP_1_MAX_SEQ_LEN,
        "summation_group": 1,
        "target_cont_ratio": 0.05,
        "class_balance": [0.41733, 0.58267, 1.0, 1.0, 1.0],
    },
    "tok_splice_junctions": {
        "type": "chain_token",
        "tokenizer": splice_junctions_tokenizer,
        "min_tokens": GROUP_1_MIN_TOK,
        "max_tokens": GROUP_1_MAX_TOK,
        "max_seq_len": GROUP_1_MAX_SEQ_LEN,
        "summation_group": 1,
        "class_balance": [0.99851, 0.0007431, 0.0007434, 1, 1, 1],
        "target_cont_ratio": 0.05,
    },
    "tok_splice_jctns_5cls": {
        "type": "chain_token",
        "tokenizer": splice_jctns_5cls_tokenizer,
        "min_tokens": GROUP_1_MIN_TOK,
        "max_tokens": GROUP_1_MAX_TOK,
        "max_seq_len": GROUP_1_MAX_SEQ_LEN,
        "summation_group": 1,
        "class_balance": [0.99826, 0.0007, 0.0007, 0.00017, 0.00017, 1.0, 1.0, 1.0],
        "target_cont_ratio": 0.05,
    },
    "tok_cds": {
        "type": "chain_token",
        "tokenizer": cds_tokenizer,
        "min_tokens": GROUP_1_MIN_TOK,
        "max_tokens": GROUP_1_MAX_TOK,
        "max_seq_len": GROUP_1_MAX_SEQ_LEN,
        "summation_group": 1,
        "target_cont_ratio": 0.05,
        "class_balance": [0.41174, 0.58826, 1.0, 1.0, 1.0],
    },
    "tok_cds_junctions": {
        "type": "chain_token",
        "tokenizer": cds_junctions_tokenizer,
        "min_tokens": GROUP_1_MIN_TOK,
        "max_tokens": GROUP_1_MAX_TOK,
        "max_seq_len": GROUP_1_MAX_SEQ_LEN,
        "summation_group": 1,
        "class_balance": [0.99843, 0.00079, 0.00079, 1.0, 1.0, 1.0],
        "target_cont_ratio": 0.05,
    },
    "tok_utr": {
        "type": "chain_token",
        "tokenizer": utr_tokenizer,
        "min_tokens": GROUP_1_MIN_TOK,
        "max_tokens": GROUP_1_MAX_TOK,
        "max_seq_len": GROUP_1_MAX_SEQ_LEN,
        "summation_group": 1,
        "target_cont_ratio": 0.05,
    },
    # GROUP 1: DNA MODALITIES
    "tok_phylop_mouse": {
        "type": "chain_token",
        "tokenizer": phylop_mouse_tokenizer,
        "min_tokens": GROUP_1_MIN_TOK,
        "max_tokens": GROUP_1_MAX_TOK,
        "max_seq_len": GROUP_1_MAX_SEQ_LEN,
        "summation_group": 1,
        "target_cont_ratio": 0.05,
    },
    "tok_phylop_human": {
        "type": "chain_token",
        "tokenizer": phylop_human_tokenizer,
        "min_tokens": GROUP_1_MIN_TOK,
        "max_tokens": GROUP_1_MAX_TOK,
        "max_seq_len": GROUP_1_MAX_SEQ_LEN,
        "summation_group": 1,
        "target_cont_ratio": 0.05,
    },
    "tok_atac": {
        "type": "chain_token",
        "tokenizer": atac_tokenizer,
        "min_tokens": GROUP_1_MIN_TOK,
        "max_tokens": GROUP_1_MAX_TOK,
        "max_seq_len": GROUP_1_MAX_SEQ_LEN,
        "summation_group": 1,
        "target_cont_ratio": 0.05,
        "class_balance": [0.0, 0.0013, 0.00182, 0.00223, 0.00265, 0.0032, 0.004, 0.00546, 0.00809, 0.01299, 0.93845, 0.0198, 1.0, 1.0, 1.0],
    },
    # UNGROUPED MODALITIES (they are their own group)
    "tok_kingdom": {
        "type": "chain_token",
        "tokenizer": kingdom_tokenizer,
        "min_tokens": 0,
        "max_tokens": 1,
        "max_seq_len": 1,
        "summation_group": 2,
        "class_balance": [0.33899, 0.04261, 0.0, 0.03234, 0.03587, 0.0, 0.0, 0.4977, 0.0, 0.0, 0.02485, 1.0, 1.0, 0.02763],
    },
    "tok_is_coding": {
        "type": "chain_token",
        "tokenizer": is_coding_tokenizer,
        "min_tokens": 0,
        "max_tokens": 1,
        "max_seq_len": 1,
        "summation_group": 3,
        "class_balance": [0.03948, 0.96052, 1.0, 1.0, 1.0],
    },
    "tok_funcprot_caption": {
        "tokenizer": text_tokenizer,
        "min_tokens": 0,
        "max_tokens": 'context_len',
        "max_seq_len": 'context_len',
        "type": "text_token_all_targets",
        "summation_group": 4,
    },
    "tok_prot_abund": {
        "type": "chain_token",
        "tokenizer": prot_abund_tokenizer,
        "min_tokens": 0,
        "max_tokens": 1,
        "max_seq_len": 1,
        "summation_group": 5,
    },
    "tok_feature_type": {
        "type": "chain_token",
        "tokenizer": feature_type_tokenizer,
        "min_tokens": 0,
        "max_tokens": 1,
        "max_seq_len": 1,
        "summation_group": 6,
        "class_balance": [0.01073, 0.0036, 0.00074, 0.95717, 0.02294, 0.0001, 0.00099, 1.0, 1.0, 1.0],
    }   
}

MODALITY_INFO['tok_context'] = {
    "type": MODALITY_INFO['tok_funcprot_caption']['type'],
    "tokenizer": MODALITY_INFO['tok_funcprot_caption']['tokenizer'],
    "min_tokens": MODALITY_INFO['tok_funcprot_caption']['min_tokens'],
    "max_tokens": MODALITY_INFO['tok_funcprot_caption']['max_tokens'],
    "max_seq_len": MODALITY_INFO['tok_funcprot_caption']['max_seq_len'],
    "summation_group": 7,
}

MODALITY_INFO['tok_corpus'] = {
    "type": MODALITY_INFO['tok_funcprot_caption']['type'],
    "tokenizer": MODALITY_INFO['tok_funcprot_caption']['tokenizer'],
    "min_tokens": MODALITY_INFO['tok_funcprot_caption']['min_tokens'],
    "max_tokens": MODALITY_INFO['tok_funcprot_caption']['max_tokens'],
    "max_seq_len": MODALITY_INFO['tok_funcprot_caption']['max_seq_len'],
    "summation_group": 8,
}

MODALITY_INFO['tok_gene_family_txt'] = {
    "type": MODALITY_INFO['tok_funcprot_caption']['type'],
    "tokenizer": MODALITY_INFO['tok_funcprot_caption']['tokenizer'],
    "min_tokens": MODALITY_INFO['tok_funcprot_caption']['min_tokens'],
    "max_tokens": MODALITY_INFO['tok_funcprot_caption']['max_tokens'],
    "max_seq_len": MODALITY_INFO['tok_funcprot_caption']['max_seq_len'],
    "summation_group": 9,
}

partial_init_embeddings(MODALITY_INFO)

mod_groups = {
    g: [k for k, v in MODALITY_INFO.items() if v["summation_group"] == g]
    for g in set(v["summation_group"] for v in MODALITY_INFO.values())
}
group_names = {
    0: GROUP_0_NAME,
    1: GROUP_1_NAME,
    generate_uint15_hash("register"): "register",
}


for mod, d in MODALITY_INFO.items():
    if d["summation_group"] not in group_names:
        group_names[d["summation_group"]] = mod.split("tok_")[-1]

GROUP_INFO = {}
for id, name in group_names.items():
    mods = {mod for mod, info in MODALITY_INFO.items() if info["summation_group"] == id}
    max_seq_lens = {MODALITY_INFO[mod]["max_seq_len"] for mod in mods}
    assert len(max_seq_lens) <= 1, (
        f"Group {name} has multiple max_seq_len values: {max_seq_lens}"
    )
    max_seq_len = max_seq_lens.pop() if max_seq_lens else None
    GROUP_INFO[id] = {"name": name, "mods": mods, "max_seq_len": max_seq_len}


# ---------------------------------------------------------------------------
# Pathway gating
#
# MIMIC is trained on specific (conditioning -> target) combinations. By default
# generate() only allows the well-supported ones and RAISES on anything else,
# telling the caller how to override (pass on_unsupported="allow" to run it, or
# "warn" to run with a warning). A requested target T is supported when some
# staged input i reaches it via one of:
#   * within-track  -- i and T are in the same molecular track (protein / nucleic);
#   * text association -- a text modality paired with a modality it was trained
#     with (TEXT_ASSOCIATIONS).
#
# Cross-track pathways are NOT enabled by default (PATHWAY_ALLOWLIST is empty).
# Tracks come from the two summation groups (protein=0, nucleic=1) plus the
# singleton assignments in _SINGLETON_TRACK; "text" modalities are the free-text
# channels. Selectively enable a cross-track pathway by adding an (input, target)
# pair to PATHWAY_ALLOWLIST.
# ---------------------------------------------------------------------------

# Package-wide default when generate() is called without an explicit
# `on_unsupported`. "error" (raise), "warn" (loud loguru warning, still runs), or
# "allow" (silent). Set to "warn"/"allow" for a looser install.
DEFAULT_ON_UNSUPPORTED = "error"

_TRACK_BY_GROUP = {0: "protein", 1: "nucleic"}

# CURATED: track of each singleton (non group-0/1) modality.
_SINGLETON_TRACK = {
    "tok_prot_abund": "protein",
    "tok_funcprot_caption": "protein",   # protein functional caption
    "tok_is_coding": "nucleic",
    "tok_feature_type": "nucleic",
    "tok_context": "text",
    "tok_corpus": "text",
    "tok_gene_family_txt": "text",
    "tok_kingdom": "text",
}

MODALITY_TRACK = {
    mod: _TRACK_BY_GROUP.get(
        info["summation_group"], _SINGLETON_TRACK.get(mod, "unknown")
    )
    for mod, info in MODALITY_INFO.items()
}

# CURATED: text modality <-> modalities it was trained jointly with. Extends text
# across tracks (within-track text pairs are already covered by MODALITY_TRACK).
TEXT_ASSOCIATIONS = {
    "tok_context": {"tok_atac", "tok_cage", "tok_rasp2", "tok_prot_abund"},
}

# CURATED HOOK: explicit cross-track (input -> target) pathways allowed beyond
# within-track + text. Add (input, target) pairs here to selectively enable more
# pathways. Empty by default: the supported gate is within-track + text-association.
PATHWAY_ALLOWLIST = set()


def _pair_supported(i, t):
    """Is generating target `t` from a single conditioning input `i` a supported pathway?"""
    ti, tt = MODALITY_TRACK.get(i, "unknown"), MODALITY_TRACK.get(t, "unknown")
    if ti == tt and ti in ("protein", "nucleic"):
        return True
    for txt, partners in TEXT_ASSOCIATIONS.items():
        if (i == txt and t in partners) or (t == txt and i in partners):
            return True
    return (i, t) in PATHWAY_ALLOWLIST


def unsupported_pathways(input_mods, target_mods):
    """Return {target: reason} for requested targets with no supported pathway.

    A target is supported when some staged input reaches it via a within-track,
    text-association, or PATHWAY_ALLOWLIST pathway (see the module header). Args
    map to model modality keys (e.g. "tok_aa_seq").
    """
    input_mods = set(input_mods)
    in_tracks = sorted({MODALITY_TRACK.get(i, "unknown") for i in input_mods}) or ["none"]
    unsupported = {}
    for t in target_mods:
        if not any(_pair_supported(i, t) for i in input_mods):
            tt = MODALITY_TRACK.get(t, "unknown")
            unsupported[t] = f"{tt}-track target not reachable from inputs (tracks: {in_tracks})"
    return unsupported


def build_modality_info(in_domains, out_domains, num_input_tokens):
    """Reconstruct the per-run modality_info / group_info from the global registry.

    Replaces the training-time `setup_data`/`setup_mod_group_info` path: given only
    the model's input/output domains and token budget (from the checkpoint config),
    it returns the filtered `modality_info` and `group_info` needed to build and run
    the model — no data configs or dataloaders required.

    `'context_len'` placeholders for token budgets are resolved to `num_input_tokens`.
    Returns shallow copies so the global MODALITY_INFO/GROUP_INFO are never mutated.
    """
    all_domains = sorted(set(in_domains) | set(out_domains))
    missing = [m for m in all_domains if m not in MODALITY_INFO]
    if missing:
        raise KeyError(f"Unknown modalities not in the registry: {missing}")

    modality_info = {}
    for m in all_domains:
        info = dict(MODALITY_INFO[m])
        for key in ("min_tokens", "max_tokens", "max_seq_len"):
            if info.get(key) == "context_len":
                info[key] = num_input_tokens
        modality_info[m] = info

    group_info = {}
    for gid, g in GROUP_INFO.items():
        mods = {m for m in g["mods"] if m in all_domains}
        if not mods:
            continue
        max_seq_len = g["max_seq_len"]
        if max_seq_len == "context_len":
            max_seq_len = num_input_tokens
        group_info[gid] = {"name": g["name"], "mods": mods, "max_seq_len": max_seq_len}

    return modality_info, group_info
