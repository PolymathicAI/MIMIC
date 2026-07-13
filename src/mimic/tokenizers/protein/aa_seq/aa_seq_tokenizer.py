from ...base import CharLevelTokenizer
from loguru import logger

AMINO_VOCAB = ["L", "A", "G", "V", "S", "E", "R", "T", "I", "D", "P", "K", "Q", "N", "F", "Y", "M", "H", "W", "C", "B", "U", "Z", "O"]
# L: Leucine
# A: Alanine
# G: Glycine
# V: Valine
# S: Serine
# E: Glutamic Acid
# R: Arginine
# T: Threonine
# I: Isoleucine
# D: Aspartic Acid
# P: Proline
# K: Lysine
# Q: Glutamine
# N: Asparagine
# F: Phenylalanine
# Y: Tyrosine
# M: Methionine
# H: Histidine
# W: Tryptophan
# C: Cysteine
# B: Asx (Asparagine or Aspartic Acid)
# U: Sec (Selenocysteine)
# Z: Glx (Glutamine or Glutamic Acid)
# O: Pyl (Pyrrolysine)
# X: Unknown (not in the standard 20 amino acids)

_version = "1.0"

aa_seq_tokenizer = CharLevelTokenizer(vocab_list=AMINO_VOCAB, unk_token="X", version=_version)

# %%
# run a bunch of tests with randomly generated DNA sequences


def test_tokenizer(tokenizer, length=100):

    import random

    vocab = [el for el in tokenizer.vocab_list if 
             el not in [tokenizer.pad_token, tokenizer.mask_token, tokenizer.unk_token]]

    def random_sample(length):
        return "".join(random.choices(vocab, k=length))

    for _ in range(100):
        seq = random_sample(length)
        tokens = tokenizer.tokenize(seq)
        decoded = tokenizer.detokenize(tokens)
        assert seq == decoded, f"Failed on sequence: {seq} -> {decoded}"


if __name__ == "__main__":
    test_tokenizer(tokenizer=aa_seq_tokenizer)
