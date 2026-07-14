# %%

import os

import numpy as np

from loguru import logger

from ...base import DigitizeTokenizer

_version = "1.3"
_data_version = "1.1"
_center_type = "weighted_average"

prot_abund_data = np.load(
    os.path.join(
        os.path.dirname(__file__), f"prot_abund_bin_centers_{_data_version}.npz"
    )
)

prot_abund_tokenizer = DigitizeTokenizer(
    bins=prot_abund_data["bins"],
    centers=prot_abund_data[_center_type].tolist(),
    version=_version,
    data_version=_data_version,
)

# %%
# run a bunch of tests with randomly generated numbers


def test_tokenizer(tokenizer, length=1000):
    prot_abund_bins = prot_abund_data["bins"]

    def random_sample(length):
        return np.random.uniform(
            min(prot_abund_bins), max(prot_abund_bins), size=length
        )

    bin_sizes = np.diff(prot_abund_bins)

    for _ in range(100):
        seq = random_sample(length)
        tokens = tokenizer.tokenize(seq)

        # get the bin size for each token
        max_error = (1 + 0.01) * bin_sizes[
            tokens
        ]  # a little bit of wiggle room for rounding errors

        decoded = tokenizer.detokenize(tokens)
        assert all(np.abs(np.array(seq) - np.array(decoded)) <= max_error), (
            f"Failed on sequence: {seq} -> {decoded}"
        )


if __name__ == "__main__":
    test_tokenizer(tokenizer=prot_abund_tokenizer)
