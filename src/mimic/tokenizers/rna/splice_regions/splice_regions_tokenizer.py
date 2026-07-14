# %%

from ...base import CharLevelTokenizer

_version = "1.1"

splice_regions_tokenizer = CharLevelTokenizer(
    vocab_list=['0', '1'],  # Boolean values
    version=_version
)

# %%
# run a bunch of tests with randomly generated numbers
def test_tokenizer(tokenizer, length=1000):
    import random

    def random_sample(length):
        return ''.join(str(random.choice([0, 1])) for _ in range(length))

    for _ in range(100):
        seq = random_sample(length)
        tokens = tokenizer.tokenize(seq)
        decoded = tokenizer.detokenize(tokens)
        assert seq == decoded, \
            f"Failed on sequence: {seq} -> {decoded}"


if __name__ == "__main__":
    test_tokenizer(tokenizer=splice_regions_tokenizer)
