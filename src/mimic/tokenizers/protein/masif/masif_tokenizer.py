# %%

import numpy as np
import os
from ...base import BaseTokenizer

_version = "1.3"
_data_version = "v4_10m"
_center_type = "weighted_average"

# List of MASIF features
MASIF_FEATURES = ["n_vertices", "si_index", "charge", "hbond", "hydrophobicity"]


class MasifTokenizer(BaseTokenizer):
    """
    We define a custom tokenizer for MASIF features because the nan values are meaningful and should be preserved during training.
    NaN values meanes that the residue is buried in the bulk with no surface exposure.
    """

    def __init__(
        self,
        bins: list[float],
        version: str,
        data_version: str = None,
        centers: list[float] = None,
    ):
        self.pad_token = "[PAD]"
        self.mask_token = "[MASK]"
        self.unk_token = np.nan
        self.buried_token = np.nan
        self.version = str(version)
        if data_version is not None:
            self.data_version = str(data_version)

        # load the digitizer bins from numpy array
        self.bins = [float(el) for el in bins]

        # assert that the bins are sorted
        assert all(x < y for x, y in zip(self.bins[:-1], self.bins[1:])), (
            "Bins must be strictly sorted",
        )

        # Add/Subtract 1e-6 from endpoints to make them inclusive
        self.bins[-1] += 1e-6
        self.bins[0] -= 1e-6

        if centers is None:
            centers = [(x + y) / 2 for x, y in zip(self.bins[:-1], self.bins[1:])]
        else:
            assert len(centers) == len(self.bins) - 1, (
                "Centers length must be equal to bins length - 1"
            )
            assert all(x < y for x, y in zip(centers[:-1], centers[1:])), (
                "Centers must be strictly sorted"
            )

        # define the center of the bins as the vocab for detokenization
        self.vocab_list = (
            centers
            + [self.buried_token]
            + [self.pad_token, self.mask_token, self.unk_token]
        )
        self.vocab_size = len(self.vocab_list)

        self.token_to_id = {tok: i for i, tok in enumerate(self.vocab_list)}
        self.id_to_token = {i: tok for tok, i in self.token_to_id.items()}

        self.buried_token_id = self.token_to_id[self.buried_token]
        self.pad_token_id = self.token_to_id[self.pad_token]
        self.mask_token_id = self.token_to_id[self.mask_token]
        self.unk_token_id = self.token_to_id[self.unk_token]

        super().__init__(has_weighted_mean=True)

    def tokenize(self, x: list[float]) -> list[int]:
        # convert input to numpy array
        x = np.asarray(x, dtype=np.float64).reshape(-1)

        # create mask for values outside bin range
        out_of_range = (x < self.bins[0]) | (x > self.bins[-1])

        # digitize the values
        digitized = np.digitize(x, self.bins, right=False) - 1

        # replace out of range values with unk_token_id
        digitized = np.where(out_of_range, self.unk_token_id, digitized)

        # replace NaN values with buried_token_id
        digitized = np.where(np.isnan(x), self.buried_token_id, digitized)

        # convert back to list of integers
        return digitized.astype(int).tolist()

    def detokenize(
        self, x: list[int] | None = None, probs: np.ndarray | None = None
    ) -> list[float]:
        if probs is None:
            if x is None:
                raise ValueError("x is required when probs is not provided.")
            return [self.id_to_token[i] for i in x]
            
        n_pos = probs.shape[0]
        if x is not None:
            x_arr = np.asarray(x)
            if len(x_arr) != n_pos:
                raise ValueError(
                    f"len(x)={len(x_arr)} must equal probs.shape[0]={n_pos} when both are provided."
                )
        
        num_bins = self.vocab_size - 4
        assert probs.shape in (
            (n_pos, self.vocab_size),
            (n_pos, num_bins),
        ), (
            f"probs shape {probs.shape} must be (n_pos={n_pos}, vocab_size={self.vocab_size}) "
            f"or (n_pos={n_pos}, vocab_size-4={num_bins})"
        )
        
        n_probs = min(probs.shape[1], self.vocab_size)
        result = np.zeros(n_pos)
        for pos in range(n_pos):
            total = 0.0
            for i in range(n_probs):
                if i not in self.id_to_token:
                    continue
                val = self.id_to_token[i]
                if isinstance(val, (int, float)) and not np.isnan(val):
                    total += probs[pos, i] * float(val)
            result[pos] = total
        return result.tolist()


# Load bins and create tokenizers for each feature
masif_tokenizers = {}

for feature in MASIF_FEATURES:
    feature_data = np.load(
        os.path.join(
            os.path.dirname(__file__),
            f"masif_{feature}_bin_centers_{_data_version}.npz",
        )
    )

    masif_tokenizers[feature] = MasifTokenizer(
        bins=feature_data["bins"],
        centers=feature_data[_center_type].tolist(),
        version=_version,
        data_version=_data_version,
    )

# Create individual tokenizers for easy access
n_vertices_tokenizer = masif_tokenizers["n_vertices"]
si_index_tokenizer = masif_tokenizers["si_index"]
charge_tokenizer = masif_tokenizers["charge"]
hbond_tokenizer = masif_tokenizers["hbond"]
hydrophobicity_tokenizer = masif_tokenizers["hydrophobicity"]

# %%
# run a bunch of tests with randomly generated numbers


def test_tokenizer(tokenizer, bins, feature_name, length=1000):
    def random_sample(length, bins):
        return np.random.uniform(min(bins), max(bins), size=length)

    bin_sizes = np.diff(bins)

    for _ in range(100):
        seq = random_sample(length, bins)
        tokens = tokenizer.tokenize(seq)
        # get the bin size for each token
        max_error = (1 + 0.01) * bin_sizes[
            tokens
        ]  # a little bit of wiggle room for rounding errors
        decoded = tokenizer.detokenize(tokens)
        assert all(np.abs(np.array(seq) - np.array(decoded)) <= max_error), (
            f"Failed on sequence: {seq} -> {decoded}"
        )

    # Test with NaN values
    for _ in range(100):
        seq = random_sample(length, bins)
        # Randomly insert NaN values
        nan_indices = np.random.choice(length, size=length // 10, replace=False)
        seq[nan_indices] = np.nan

        tokens = tokenizer.tokenize(seq)
        decoded = tokenizer.detokenize(tokens)
        decoded_arr = np.array(decoded)
        seq_arr = np.array(seq)

        # Check that NaN values are preserved
        assert np.array_equal(np.isnan(seq_arr), np.isnan(decoded_arr)), (
            f"NaN positions don't match: original NaN at {np.where(np.isnan(seq_arr))}, decoded NaN at {np.where(np.isnan(decoded_arr))}"
        )

        # Check non-NaN values are within tolerance
        non_nan_mask = ~np.isnan(seq_arr)
        if np.any(non_nan_mask):
            non_nan_tokens = np.array(tokens)[non_nan_mask]
            max_error = (1 + 0.01) * bin_sizes[non_nan_tokens]
            assert all(
                np.abs(seq_arr[non_nan_mask] - decoded_arr[non_nan_mask]) <= max_error
            ), "Failed on non-NaN values"

def test_tokenizer_probs(tokenizer, feature_name):
    n_pos = 4
    vocab_size = tokenizer.vocab_size
    probs = np.zeros((n_pos, vocab_size))
    
    # 1. 100% valid bin
    valid_idx = 10
    probs[0, valid_idx] = 1.0
    exp_0 = float(tokenizer.id_to_token[valid_idx])
    
    # 2. 50/50 valid bins
    valid_idx2 = 20
    probs[1, valid_idx] = 0.5
    probs[1, valid_idx2] = 0.5
    exp_1 = 0.5 * float(tokenizer.id_to_token[valid_idx]) + 0.5 * float(tokenizer.id_to_token[valid_idx2])
    
    # 3. 50% valid bin, 50% NaN (buried) token
    probs[2, valid_idx] = 0.5
    probs[2, tokenizer.buried_token_id] = 0.5
    # The NaN value is skipped, so the total will be just the 0.5 weight of the valid token
    exp_2 = 0.5 * float(tokenizer.id_to_token[valid_idx])
    
    # 4. 100% NaN (buried) token
    probs[3, tokenizer.buried_token_id] = 1.0
    exp_3 = 0.0
    
    decoded = tokenizer.detokenize(probs=probs)
    
    assert np.isclose(decoded[0], exp_0), f"Fail 100% valid: expected {exp_0}, got {decoded[0]}"
    assert np.isclose(decoded[1], exp_1), f"Fail 50/50 valid: expected {exp_1}, got {decoded[1]}"
    assert np.isclose(decoded[2], exp_2), f"Fail 50/50 valid/NaN: expected {exp_2}, got {decoded[2]}"
    assert np.isclose(decoded[3], exp_3), f"Fail 100% NaN: expected {exp_3}, got {decoded[3]}"


if __name__ == "__main__":
    # Test all tokenizers
    for feature in MASIF_FEATURES:
        feature_data = np.load(
            os.path.join(
                os.path.dirname(__file__),
                f"masif_{feature}_bin_centers_{_data_version}.npz",
            )
        )
        feature_bins = feature_data["bins"]
        test_tokenizer(
            tokenizer=masif_tokenizers[feature], bins=feature_bins, feature_name=feature
        )
        test_tokenizer_probs(
            tokenizer=masif_tokenizers[feature], feature_name=feature
        )
