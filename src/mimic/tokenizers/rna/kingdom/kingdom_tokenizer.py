
from ...base import ClassTokenizer
"""
This module provides tokenization utilities for RNA kingdom classification.

Attributes:
    KINGDOM_VOCAB (list): List of recognized biological kingdoms for RNA data.

"""

_version = "1.0"

KINGDOM_VOCAB = ['Pseudomonadati', 'Bacillati', 'Fungi', 'Metazoa', 'Thermotogati',
       'Viridiplantae', 'Thermoproteati', 'Methanobacteriati',
       'Fusobacteriati', 'Nanobdellati', 'Promethearchaeati']

# - 'Pseudomonadati': A diverse group of bacteria, including many important pathogens.
# - 'Bacillati': Bacteria characterized by rod-shaped cells, includes Bacillus species.
# - 'Fungi': Eukaryotic organisms including yeasts, molds, and mushrooms.
# - 'Metazoa': Multicellular animals.
# - 'Thermotogati': Thermophilic bacteria with distinctive sheath-like outer membranes.
# - 'Viridiplantae': Green plants, including land plants and green algae.
# - 'Thermoproteati': A class of thermophilic archaea.
# - 'Methanobacteriati': Methanogenic archaea, producing methane as a metabolic byproduct.
# - 'Fusobacteriati': Anaerobic, Gram-negative bacteria, some pathogenic.
# - 'Nanobdellati': Small, predatory bacteria.
# - 'Promethearchaeati': Recently described group of archaea, related to the origin of eukaryotes.

kingdom_tokenizer = ClassTokenizer(vocab_list=KINGDOM_VOCAB, version=_version)

# %%
# run a bunch of tests with randomly generated numbers

def test_tokenizer(tokenizer, length=20):
    import random

    def random_sample(length):
        sample = [random.choice(KINGDOM_VOCAB) for _ in range(length)]

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
    test_tokenizer(tokenizer=kingdom_tokenizer)
