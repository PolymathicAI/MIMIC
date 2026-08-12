"""Provide a standard interface for filtering sequences and structures on things like length, quality, and plDDT so we have a consistent way to ensure high quality data."""

# %%
from __future__ import annotations

import typing as T
from abc import ABC

import biotite.structure as bs
import numpy as np

if T.TYPE_CHECKING:
    # Only referenced in annotations, so `lore` does not depend on torch at runtime.
    import torch

# %%
filters = {}


class SequenceFilter(ABC):
    def __init__(self, name):
        self.name = name

    def __call__(self, seq: T.Iterable[str], *args, **kwargs):
        return [s for s in seq if self.check(s, *args, **kwargs)]

    def __repr__(self):
        return f"<SequenceFilter: {self.name}>"

    def __and__(self, other):
        return ComposeFilter([self, other])

    def _filter(self, seq: str) -> bool:
        raise NotImplementedError

    def check(self, seq: str, *args, **kwargs):
        assert isinstance(seq, str), f"Expected str, got {type(seq)}"
        return self._filter(seq, *args, **kwargs)


class StructureFilter(ABC):
    def __init__(self, name):
        self.name = name

    def __call__(
        self, struct: T.Iterable[bs.AtomArray] | bs.AtomArrayStack, *args, **kwargs
    ):
        return [s for s in struct if self.check(s, *args, **kwargs)]

    def __repr__(self):
        return f"<StructureFilter: {self.name}>"

    def __and__(self, other):
        return ComposeFilter([self, other])

    def _filter(self, struct: bs.AtomArray) -> bool:
        raise NotImplementedError

    def check(self, struct: bs.AtomArray, *args, **kwargs):
        assert isinstance(struct, bs.AtomArray), (
            f"Expected AtomArray, got {type(struct)}"
        )
        return self._filter(struct, *args, **kwargs)


class TokenFilter(ABC):
    def __init__(
        self,
        name,
    ):
        self.name = name

    def __call__(
        self, tokens: T.Iterable[list[int]] | np.ndarray | torch.tensor, *args, **kwargs
    ):
        return [s for s in tokens if self.check(s, *args, **kwargs)]

    def __repr__(self):
        return f"<TokenFilter: {self.name}>"

    def __and__(self, other):
        return ComposeFilter([self, other])

    def _filter(self, tokens: list[int] | np.ndarray | torch.Tensor) -> bool:
        raise NotImplementedError

    def check(self, tokens: list[int] | np.ndarray | torch.Tensor, *args, **kwargs):
        if isinstance(tokens, list):
            tokens = np.array(tokens)
        assert tokens.shape[0] <= 1, (
            f"Expected only a single sample, both batch of size {len(tokens)}"
        )

        return self._filter(tokens, *args, **kwargs)


class ComposeFilter:
    def __init__(self, filters):
        self.filters = filters

    def __call__(self, batch, *args, **kwargs):
        for f in self.filters:
            batch = f(batch, *args, **kwargs)
        return batch

    def __repr__(self):
        return f"<ComposeFilter: {self.filters}>"

    def check(self, inp, *args, **kwargs):
        return all(f.check(inp, *args, **kwargs) for f in self.filters)


# %%
class MinLengthFilter(SequenceFilter):
    def __init__(self, min_length):
        super().__init__("MinLengthFilter")
        self.min_length = min_length

    def _filter(self, seq):
        return len(seq) >= self.min_length


class MaxLengthFilter(SequenceFilter):
    def __init__(self, max_length):
        super().__init__("MaxLengthFilter")
        self.max_length = max_length

    def filter(self, seq):
        return len(seq) <= self.max_length


class CodonLengthFilter(SequenceFilter):
    def __init__(self):
        super().__init__("CodonLengthFilter")

    def _filter(self, seq):
        return len(seq) % 3 == 0


class StartCodonFilter(SequenceFilter):
    def __init__(self):
        super().__init__("StartCodonFilter")

    def _filter(self, seq):
        return seq[:3] in ["ATG", "GTG", "TTG"]


class StopCodonFilter(SequenceFilter):
    def __init__(self):
        super().__init__("StopCodonFilter")

    def _filter(self, seq):
        return seq[-3:] in ["TAA", "TAG", "TGA"]


class pLDDTFilter(StructureFilter):
    def __init__(self, min_pLDDT):
        super().__init__("pLDDTFilter")
        self.min_pLDDT = min_pLDDT

    def _filter(self, struct: bs.AtomArray):
        try:
            return np.mean(struct.get_annotation("b_factor")) >= self.min_pLDDT
        except ValueError:
            raise ValueError("No B-factors found in structure")
