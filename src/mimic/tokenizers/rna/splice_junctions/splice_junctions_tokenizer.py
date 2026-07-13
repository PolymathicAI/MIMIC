from ...base import CharLevelTokenizer
from loguru import logger

_version = "1.0"

JUNCTION_VOCAB= ["0", "1", "2"]
# 0: No junction
# 1: Acceptor junction
# 2: Donor junction

splice_junctions_tokenizer = CharLevelTokenizer(vocab_list=JUNCTION_VOCAB, version=_version)

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
    test_tokenizer(tokenizer=splice_junctions_tokenizer)
