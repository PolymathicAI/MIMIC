
from ...base import ClassTokenizer
"""
This module provides tokenization utilities for RNA feature type classification.
"""

_version = "1.0"

FEATURE_TYPE_VOCAB = ['lncRNA', 'miscRNA', 'other', 'protein_coding', 'pseudogene', 'rRNA', 'tRNA']

# - 'lncRNA': Long non-coding RNA, involved in gene regulation.
# - 'miscRNA': Miscellaneous RNA, a category for various non-coding RNAs.
# - 'other': Any RNA type that does not fit into the predefined categories.
# - 'protein_coding': RNA that codes for proteins.
# - 'pseudogene': Non-functional sequences similar to known genes.
# - 'rRNA': Ribosomal RNA, a component of ribosomes essential for protein synthesis.
# - 'tRNA': Transfer RNA, involved in translating mRNA into amino acids during protein synthesis.

feature_type_tokenizer = ClassTokenizer(vocab_list=FEATURE_TYPE_VOCAB, version=_version)

# %%
# run a bunch of tests with randomly generated numbers

def test_tokenizer(tokenizer, length=20):
    import random

    def random_sample(length):
        sample = [random.choice(FEATURE_TYPE_VOCAB) for _ in range(length)]

        # if length is 1, pass just the value, not a list
        sample = sample[0] if length == 1 else sample
        return sample
    

    for _ in range(100):
        seq = random_sample(length)
        tokens = tokenizer.tokenize(seq)
        decoded = tokenizer.detokenize(tokens)
        assert seq == decoded, \
            f"Failed on sequence: {seq} -> {decoded}"


if __name__ == "__main__":
    test_tokenizer(tokenizer=feature_type_tokenizer)


