from ...base import CharLevelTokenizer
from loguru import logger
import numpy as np

_version = "1.0"

ATAC_VOCAB = ["N", "X", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
# The vocab are the quantiles with 0 corresponding to no peak.
        

atac_tokenizer = CharLevelTokenizer(vocab_list=ATAC_VOCAB, version=_version, unk_token="U")

# %%
# run a bunch of tests with randomly generated sequences

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
    test_tokenizer(tokenizer=atac_tokenizer)
