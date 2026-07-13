# %%

from ...base import BoolTokenizer

_version = "1.0"

is_coding_tokenizer = BoolTokenizer(version=_version)

# %%
# run a bunch of tests with randomly generated numbers
def test_tokenizer(tokenizer, length=1000):
    import random

    def random_sample(length):
        return [random.choice([True, False]) for _ in range(length)]

    for _ in range(100):
        seq = random_sample(length)
        tokens = tokenizer.tokenize(seq)
        decoded = tokenizer.detokenize(tokens)
        assert seq == decoded, \
            f"Failed on sequence: {seq} -> {decoded}"


if __name__ == "__main__":
    test_tokenizer(tokenizer=is_coding_tokenizer)
