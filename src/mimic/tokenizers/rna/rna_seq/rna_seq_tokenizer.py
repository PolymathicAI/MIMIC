from ...base import CharLevelTokenizer
from loguru import logger

_version = "1.0"

RNA_VOCAB = ["A", "C", "G", "U"]
# A: Adenine
# C: Cytosine
# G: Guanine
# T: Thymine (replaced with U in RNA)
# U: Uracil
# N: Any base (added as an unknown token)

class RNASeqTokenizer(CharLevelTokenizer):
    def preprocess(self, x):
        # replace T with U for RNA sequences and change to uppercase
        return x.upper().replace("T", "U") if isinstance(x, str) else [el.upper().replace("T", "U") for el in x]

rna_seq_tokenizer = RNASeqTokenizer(vocab_list=RNA_VOCAB, unk_token="N", version=_version)
logger.info("The RNA tokenizer treats U and T the same in order to deal with both RNA and DNA. It cannot handle cases where both U and T appear in the same sequence.")

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
    test_tokenizer(tokenizer=rna_seq_tokenizer)
