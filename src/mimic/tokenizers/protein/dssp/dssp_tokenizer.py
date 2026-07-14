from ...base import CharLevelTokenizer
from loguru import logger

DSSP_VOCAB = ["H", "E", "C", "G", "I", "B", "T", "S", "L"]
# H: Alpha helix
# E: Extended strand
# C: Coil
# G: 3-helix (310 helix)
# I: 5-helix
# B: Beta bridge
# T: Turn
# S: Bend
# L: Loop
# X: Unknown (added as unknown token)

_version = "1.0"

dssp_tokenizer = CharLevelTokenizer(vocab_list=DSSP_VOCAB, unk_token="X", version=_version)

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
    test_tokenizer(tokenizer=dssp_tokenizer)
