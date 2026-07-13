# %%

import numpy as np
import os
from ...base import DigitizeTokenizer

_version = "1.4"
_human_data_version = "1.4"
_mouse_data_version = "1.4"
_human_center_type = "weighted_average"
_mouse_center_type = "weighted_average"

phylop_data_mouse = np.load(
    os.path.join(
        os.path.dirname(__file__),
        f"phylop_mouse_phylop_bin_centers_{_mouse_data_version}.npz",
    )
)
phylop_data_human = np.load(
    os.path.join(
        os.path.dirname(__file__),
        f"phylop_human_phylop_bin_centers_{_human_data_version}.npz",
    )
)

# phylop_bins_mouse = list(np.load(os.path.join(os.path.dirname(__file__), "phylop_bins_mouse.npy")))
# phylop_bins_human = list(np.load(os.path.join(os.path.dirname(__file__), "phylop_bins_human.npy")))

phylop_bins_mouse = phylop_data_mouse["bins"]
phylop_bins_human = phylop_data_human["bins"]

phylop_mouse_tokenizer = DigitizeTokenizer(
    bins=phylop_bins_mouse,
    centers=phylop_data_mouse[_mouse_center_type].tolist(),
    version=_version,
    data_version=_mouse_data_version,
)
phylop_human_tokenizer = DigitizeTokenizer(
    bins=phylop_bins_human,
    centers=phylop_data_human[_human_center_type].tolist(),
    version=_version,
    data_version=_human_data_version,
)

# %%
# run a bunch of tests with randomly generated numbers


def test_tokenizer(tokenizer, bins, length=1000):
    def random_sample(length):
        return np.random.uniform(min(bins), max(bins), size=length)

    bin_sizes = np.diff(bins)

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
    test_tokenizer(tokenizer=phylop_mouse_tokenizer, bins=phylop_bins_mouse)
    test_tokenizer(tokenizer=phylop_human_tokenizer, bins=phylop_bins_human)
