# %%

import numpy as np
import os
from ...base import DigitizeTokenizer
from loguru import logger

_version = "1.3"
_data_version = "v4_plddt_70"
_center_type = "weighted_average"

# sasa_bins = list(np.load(os.path.join(os.path.dirname(__file__), "sasa_bins.npy")))
sasa_bin_data = np.load(
    os.path.join(os.path.dirname(__file__), f"sasa_bin_centers_{_data_version}.npz")
)

sasa_tokenizer = DigitizeTokenizer(
    bins=sasa_bin_data["bins"],
    centers=sasa_bin_data[_center_type].tolist(),
    version=_version,
    data_version=_data_version,
)

# %%
# run a bunch of tests with randomly generated numbers


def test_tokenizer(tokenizer, length=1000):
    sasa_bins = sasa_bin_data["bins"]

    def random_sample(length):
        return np.random.uniform(min(sasa_bins), max(sasa_bins), size=length)

    bin_sizes = np.diff(sasa_bins)

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
    test_tokenizer(tokenizer=sasa_tokenizer)
