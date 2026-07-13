# %%

import numpy as np
import os
from ...base import DigitizeTokenizer
from loguru import logger

_version = "1.0"
_data_version = "1.4"
_center_type = "weighted_average"

cage_bin_data = np.load(
    os.path.join(os.path.dirname(__file__), f"cage_bin_centers_{_data_version}.npz")
)

cage_tokenizer = DigitizeTokenizer(
    bins=cage_bin_data["bins"],
    centers=cage_bin_data[_center_type].tolist(),
    version=_version,
    data_version=_data_version,
)

# %%
# run a bunch of tests with randomly generated numbers


def test_tokenizer(tokenizer, length=1000):
    cage_bins = cage_bin_data["bins"]

    def random_sample(length):
        return np.random.uniform(min(cage_bins), max(cage_bins), size=length)

    bin_sizes = np.diff(cage_bins)

    logger.debug("Testing cage tokenizer with random values...")
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

    logger.debug("✓ Random values test passed!")

    # Test with NaN values (important for cage which uses MaskedArrays)
    logger.debug("Testing cage tokenizer with NaN values...")
    for _ in range(100):
        seq = random_sample(length)
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

    logger.debug("✓ NaN values test passed!")

    # Test with MaskedArrays (the actual cage data format)
    logger.debug("Testing cage tokenizer with MaskedArrays...")
    for _ in range(100):
        # Create random data
        data = random_sample(length)

        # Create a random mask (mask ~10% of values)
        mask = np.random.random(length) < 0.1

        # Create MaskedArray
        seq = np.ma.MaskedArray(data=data, mask=mask, fill_value=1e20)

        # Tokenize the MaskedArray
        tokens = tokenizer.tokenize(seq)
        decoded = tokenizer.detokenize(tokens)
        decoded_arr = np.array(decoded)

        # Get unmasked values from original
        unmasked_indices = ~mask

        # Check that masked positions become NaN (unk_token)
        masked_positions_decoded = np.isnan(decoded_arr[mask])
        assert np.all(masked_positions_decoded), (
            f"Masked positions should become NaN, but got: {decoded_arr[mask]}"
        )

        # Check that unmasked positions are within tolerance
        unmasked_decoded = decoded_arr[unmasked_indices]
        unmasked_original = data[unmasked_indices]
        non_nan_mask = ~np.isnan(unmasked_decoded)  # In case some became NaN

        if np.any(non_nan_mask):
            # Get tokens for unmasked positions
            tokens_arr = np.array(tokens)
            unmasked_tokens = tokens_arr[unmasked_indices][non_nan_mask]
            max_error = (1 + 0.01) * bin_sizes[unmasked_tokens]

            errors = np.abs(
                unmasked_original[non_nan_mask] - unmasked_decoded[non_nan_mask]
            )
            assert np.all(errors <= max_error), (
                f"Unmasked values exceed tolerance. Max error: {errors.max()}, allowed: {max_error.max()}"
            )

        # Verify that number of NaN in decoded matches number of masked values
        assert np.sum(np.isnan(decoded_arr)) == np.sum(mask), (
            f"Number of NaN in decoded ({np.sum(np.isnan(decoded_arr))}) doesn't match mask count ({np.sum(mask)})"
        )

    logger.debug("✓ MaskedArray test passed!")
    logger.debug("cage tokenizer passed all tests!")


if __name__ == "__main__":
    test_tokenizer(tokenizer=cage_tokenizer)
