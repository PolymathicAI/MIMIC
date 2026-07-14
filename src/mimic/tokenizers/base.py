# %%

from abc import ABC, abstractmethod
from typing import List
import numpy as np
from pathlib import Path
from loguru import logger


class BaseTokenizer(ABC):
    def __init__(self, has_weighted_mean: bool = False):
        # make sure subclass __init__ already set these
        # make sure to run super at the end of subclass __init__
        required = [
            "vocab_list",
            "vocab_size",
            "pad_token_id",
            "mask_token_id",
            "unk_token_id",
            "version",
        ]
        missing = [attr for attr in required if not hasattr(self, attr)]
        if missing:
            raise AttributeError(
                f"{self.__class__.__name__} is missing required attributes: {missing}"
            )

        # assert that the last 3 elements are pad, mask, unk tokens in this order
        assert (
            self.pad_token_id == self.vocab_size - 3
            and self.mask_token_id == self.vocab_size - 2
            and self.unk_token_id == self.vocab_size - 1
        ), (
            f"pad ({self.pad_token_id}), mask ({self.mask_token_id}), unk ({self.unk_token_id}) token ids must be the last three indices in vocab of size {self.vocab_size}"
        )

        # assert that the rest is sorted
        assert self.vocab_list[:-3] == sorted(self.vocab_list[:-3]), (
            "Vocab list must be sorted except for the last three special tokens"
        )
        self.has_weighted_mean = has_weighted_mean
        super().__init__()

    def preprocess(self, x: str) -> str:
        return x

    def init(self):
        pass

    @abstractmethod
    def tokenize(self, sequence: str) -> List[int]:
        pass

    @abstractmethod
    def detokenize(self, tokens: List[int]) -> str:
        pass


class CharLevelTokenizer(BaseTokenizer):
    def __init__(
        self,
        vocab_list: List[str],
        version: str,
        data_version: str = None,
        unk_token: str = "?",
    ):
        self.pad_token = "#"
        self.mask_token = "_"
        self.unk_token = unk_token
        self.version = str(version)
        if data_version is not None:
            self.data_version = str(data_version)

        # assert that the special tokens are not in the vocab list
        for token in [self.pad_token, self.mask_token, self.unk_token]:
            assert token not in vocab_list, (
                f"Special token {token} must not be in the vocab list"
            )

        # add the special tokens to the vocab list and drop duplicates
        self.vocab_list = sorted(vocab_list) + [
            self.pad_token,
            self.mask_token,
            self.unk_token,
        ]

        # assert that the vocab list has no duplicates
        assert len(self.vocab_list) == len(set(self.vocab_list)), (
            "Vocab list must not contain duplicates"
        )

        self.vocab_size = len(self.vocab_list)

        self.token_to_id = {tok: i for i, tok in enumerate(self.vocab_list)}
        self.id_to_token = {i: tok for tok, i in self.token_to_id.items()}

        self.pad_token_id = self.token_to_id[self.pad_token]
        self.mask_token_id = self.token_to_id[self.mask_token]
        self.unk_token_id = self.token_to_id[self.unk_token]

        super().__init__()

    def tokenize(self, x: str) -> List[int]:
        x = self.preprocess(x)
        return [self.token_to_id.get(ch, self.unk_token_id) for ch in x]

    def detokenize(self, x: List[int]) -> str:
        return "".join(self.id_to_token[i] for i in x)


class DigitizeTokenizer(BaseTokenizer):
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

        # define the center of the bins as the vocab for detokenization
        if centers is None:
            centers = [(x + y) / 2 for x, y in zip(self.bins[:-1], self.bins[1:])]
        else:
            assert len(centers) == len(self.bins) - 1, (
                "Centers length must be equal to number of bins - 1"
            )
            assert all(x < y for x, y in zip(centers[:-1], centers[1:])), (
                "Centers must be strictly sorted"
            )

        self.vocab_list = centers + [self.pad_token, self.mask_token, self.unk_token]
        self.vocab_size = len(self.vocab_list)

        self.token_to_id = {tok: i for i, tok in enumerate(self.vocab_list)}
        self.id_to_token = {i: tok for tok, i in self.token_to_id.items()}

        self.pad_token_id = self.token_to_id[self.pad_token]
        self.mask_token_id = self.token_to_id[self.mask_token]
        self.unk_token_id = self.token_to_id[self.unk_token]

        super().__init__(has_weighted_mean=True)

    def tokenize(self, x: list[float]) -> list[int]:
        x = self.preprocess(x)

        # Handle MaskedArrays by converting masked values to NaN
        if isinstance(x, np.ma.MaskedArray):
            x = np.ma.filled(x, np.nan)
        # convert input to numpy array
        x = np.asarray(x, dtype=np.float64).reshape(-1)

        # create mask for values outside bin range
        out_of_range = (x < self.bins[0]) | (x > self.bins[-1])

        # digitize the values
        digitized = np.digitize(x, self.bins, right=False) - 1

        # replace out of range values with unk_token_id
        digitized = np.where(out_of_range | np.isnan(x), self.unk_token_id, digitized)

        # convert back to list of integers
        return digitized.astype(int).tolist()

    def detokenize(
        self, x: list[int] | None = None, probs: np.ndarray | None = None
    ) -> list[float]:
        """
        Detokenize a list of token ids into their corresponding float values.

        If `probs` is not provided, ids are simply mapped to token values (e.g., bin centers, etc.)
        corresponding to each id. If `probs` is provided, it is a probability matrix, and the output
        is computed as a weighted mean of the bin centers using these probabilities for each position.

        Args:
            x (list[int], optional): List of token ids to detokenize. Required when probs is not provided.
                When probs is provided, may be omitted; the number of positions is then taken from probs.shape[0].
            probs (np.ndarray, optional): If provided, should have shape (n, vocab_size) or
                (n, vocab_size - 3) with n = len(x) if x is given, else arbitrary.

        Returns:
            list[float]: List of detokenized float values corresponding to the input tokens.
        """
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
        num_bins = self.vocab_size - 3
        assert probs.shape in (
            (n_pos, self.vocab_size),
            (n_pos, num_bins),
        ), (
            f"probs shape {probs.shape} must be (n_pos={n_pos}, vocab_size={self.vocab_size}) "
            f"or (n_pos={n_pos}, vocab_size-3={num_bins})"
        )
        n_probs = probs.shape[1]
        result = np.zeros(n_pos)
        for pos in range(n_pos):
            total = 0.0
            for i in range(n_probs):
                val = self.id_to_token[i]
                if isinstance(val, (int, float)) and not (
                    isinstance(val, float) and np.isnan(val)
                ):
                    total += probs[pos, i] * float(val)
            result[pos] = total
        return result.tolist()


class BoolTokenizer(BaseTokenizer):
    def __init__(self, version: str, data_version: str = None):
        self.pad_token = 2
        self.mask_token = 3
        self.unk_token = 4
        self.version = str(version)
        if data_version is not None:
            self.data_version = str(data_version)

        vocab_list = [0, 1]
        self.vocab_list = sorted(vocab_list) + [
            self.pad_token,
            self.mask_token,
            self.unk_token,
        ]
        self.vocab_size = len(self.vocab_list)

        self.token_to_id = {tok: i for i, tok in enumerate(self.vocab_list)}
        self.id_to_token = {i: tok for tok, i in self.token_to_id.items()}

        self.pad_token_id = self.token_to_id[self.pad_token]
        self.mask_token_id = self.token_to_id[self.mask_token]
        self.unk_token_id = self.token_to_id[self.unk_token]

        super().__init__()

    def tokenize(self, x) -> list[int]:
        x = self.preprocess(x)
        x = np.array(x, dtype=int).reshape(-1)
        # replace out of range values with unk_token_id
        x = np.where((x < 0) | (x > 4), self.unk_token_id, x)
        return x

    def detokenize(self, x: list[int]) -> list:
        return [self.id_to_token[i] for i in x]


class ClassTokenizer(BaseTokenizer):
    def __init__(
        self,
        vocab_list: List[str],
        version: str,
        data_version: str = None,
        unk_token: str = "[UNK]",
    ):
        self.pad_token = "[PAD]"
        self.mask_token = "[MASK]"
        self.unk_token = unk_token
        self.version = str(version)
        if data_version is not None:
            self.data_version = str(data_version)

        # assert that the special tokens are not in the vocab list
        for token in [self.pad_token, self.mask_token, self.unk_token]:
            assert token not in vocab_list, (
                f"Special token {token} must not be in the vocab list"
            )

        # add the special tokens to the vocab list
        self.vocab_list = sorted(vocab_list) + [
            self.pad_token,
            self.mask_token,
            self.unk_token,
        ]

        # assert that the vocab list has no duplicates
        assert len(vocab_list) == len(set(vocab_list)), (
            "Vocab list must not contain duplicates"
        )
        self.vocab_size = len(self.vocab_list)

        self.token_to_id = {tok: i for i, tok in enumerate(self.vocab_list)}
        self.id_to_token = {i: tok for tok, i in self.token_to_id.items()}

        self.pad_token_id = self.token_to_id[self.pad_token]
        self.mask_token_id = self.token_to_id[self.mask_token]
        self.unk_token_id = self.token_to_id[self.unk_token]

        super().__init__()

    def tokenize(self, x: str) -> List[int]:
        x = self.preprocess(x)
        x = [x] if not isinstance(x, (list)) else x
        return [self.token_to_id.get(el, self.unk_token_id) for el in x]

    def detokenize(self, x: List[int]) -> str:
        return [self.id_to_token[i] for i in x]


class TextTokenizer(BaseTokenizer):
    """Init and encoding/decoding methods for a text Wordpiece tokenizer."""
    def __init__(self, version: str, data_version: str = None, unk_token: str = "[UNK]"):
        from transformers import AutoTokenizer

        # Version
        self.version = str(version)
        if data_version is not None:
            self.data_version = str(data_version)

        # Load BioBERT tokenizer (uses the default HuggingFace cache, i.e. HF_HOME).
        self.tokenizer_obj = AutoTokenizer.from_pretrained("dmis-lab/biobert-base-cased-v1.2")

        bert_vocab_size = len(self.tokenizer_obj.vocab)

        # Special tokens (match bio bert one's)
        self.pad_token = self.tokenizer_obj.pad_token  # "[PAD]"
        self.mask_token = self.tokenizer_obj.mask_token  # "[MASK]"
        self.unk_token = unk_token

        # NOTE: The following lines are here to comply with EVA BaseTokenizer asserts
        # BaseTokenizer requires pad, mask, unk to be the last 3 token IDs in the vocab
        # BioBert has vocab size of 28996
        self.pad_dummy_token = "[PAD_DUMMY]"
        self.mask_dummy_token = "[MASK_DUMMY]"
        self.unk_dummy_token = "[UNK_DUMMY]"

        # Add dummy tokens tokens to vocab at the end
        _ = self.tokenizer_obj.add_tokens([
                            self.pad_dummy_token, 
                            self.mask_dummy_token, 
                            self.unk_dummy_token
                            ])

        # Set dummy special token IDs at the end to comply with BaseTokenizer assertions:
        # pad_token_id = total_vocab_size - 3, mask_token_id = total_vocab_size - 2, unk_token_id = total_vocab_size - 1
        self.pad_token_id = bert_vocab_size # id=28996 (extra)
        self.mask_token_id = bert_vocab_size + 1  # id=28997 (extra)
        self.unk_token_id = bert_vocab_size + 2  # id=28998 (extra)

        # self.vocab_list is also created to comply with BaseTokenizer asserts
        keys = list(self.tokenizer_obj.vocab.keys())
        self.vocab_list = sorted([k for k in keys[:-3]]) + \
                          [self.pad_dummy_token, self.mask_dummy_token, self.unk_dummy_token] 
        self.vocab = self.tokenizer_obj.vocab
        self.vocab_size = len(self.vocab)
        
        super().__init__()

    def tokenize(self, x: str) -> List[int]:
        x = self.preprocess(x)
        # Disable special tokens to avoid adding [CLS] and [SEP] automatically
        # BioBERT tokenizer adds these by default, but we want raw token IDs
        return self.tokenizer_obj.encode(x, add_special_tokens=False)

    def detokenize(self, x: List[int]) -> str:
        return self.tokenizer_obj.decode(x)


# %%
# testing each class with a simple example
# These self-tests are guarded so that importing `mimic` has no side effects;
# the same checks are exercised as a proper suite in tests/test_tokenizers.py.
if __name__ == "__main__":
    char_tokenizer = CharLevelTokenizer(
        vocab_list=["A", "C", "G", "T"], unk_token="X", version=None
    )
    assert char_tokenizer.detokenize(char_tokenizer.tokenize("ACGTO")) == "ACGTX"

    digitizer = DigitizeTokenizer(bins=[0, 1, 2, 3, 4], version=None)
    out = digitizer.detokenize(digitizer.tokenize([0.5, 1.5, 2.5, 3.5, 4.5, np.nan]))
    assert np.isnan(out[-1]) and np.isnan(out[-2])
    assert (np.array(out[:-2]) - np.array([0.5, 1.5, 2.5, 3.5])).max() < 0.01

    # weighted-mean detokenize: 3 positions, non-trivial probs
    # bins [0,1,2,3,4] -> centers 0.5, 1.5, 2.5, 3.5
    tokens_w = [0, 1, 2]  # dummy tokens (ignored when probs given)
    probs_w = np.zeros((3, digitizer.vocab_size))
    probs_w[0, 0], probs_w[0, 1] = 0.5, 0.5  # expect 0.5*0.5 + 0.5*1.5 = 1.0
    probs_w[1, 2] = 1.0  # one-hot -> 2.5
    probs_w[2, 1], probs_w[2, 2] = 0.25, 0.75  # expect 0.25*1.5 + 0.75*2.5 = 2.25
    out_w = digitizer.detokenize(tokens_w, probs=probs_w)
    assert len(out_w) == 3
    expected_0 = 0.5 * digitizer.id_to_token[0] + 0.5 * digitizer.id_to_token[1]
    expected_1 = digitizer.id_to_token[2]
    expected_2 = 0.25 * digitizer.id_to_token[1] + 0.75 * digitizer.id_to_token[2]
    assert abs(out_w[0] - expected_0) < 1e-9 and abs(out_w[1] - expected_1) < 1e-9 and abs(out_w[2] - expected_2) < 1e-9
    # same result when probs has shape (len(x), vocab_size - 3)
    probs_bins_only = probs_w[:, : digitizer.vocab_size - 3]
    out_w_short = digitizer.detokenize(tokens_w, probs=probs_bins_only)
    assert abs(out_w_short[0] - expected_0) < 1e-9 and abs(out_w_short[1] - expected_1) < 1e-9 and abs(out_w_short[2] - expected_2) < 1e-9
    # x optional when probs given: same result with probs only
    out_w_no_x = digitizer.detokenize(probs=probs_w)
    assert abs(out_w_no_x[0] - expected_0) < 1e-9 and abs(out_w_no_x[1] - expected_1) < 1e-9 and abs(out_w_no_x[2] - expected_2) < 1e-9

    bool_tokenizer = BoolTokenizer(version=None)
    assert bool_tokenizer.detokenize(
        bool_tokenizer.tokenize([True, False, True, False, 5])
    ) == [1, 0, 1, 0, 4]

    class_tokenizer = ClassTokenizer(vocab_list=["hi", "bye"], unk_token="Other", version=None)
    assert class_tokenizer.detokenize(class_tokenizer.tokenize(["hi", "bye", "hello"])) == ["hi", "bye", "Other"]

