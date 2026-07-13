# %%
from ...base import ClassTokenizer  
from itertools import product  

_version = "1.0"

RNA_VOCAB  = ["A", "C", "G", "U"]
# A: Adenine
# C: Cytosine
# G: Guanine
# T: Thymine (replaced with U in RNA)
# U: Uracil

RNA_CODONS_VOCAB = [f"{a}{b}{c}" for (a,b,c) in product(RNA_VOCAB, repeat=3)]  

class RNA_Codons_Tokenizer(ClassTokenizer):

    @staticmethod
    def _process_one_seq(seq):
        """Helper to normalize (T->U, upper) and split one sequence."""
        # 1. Normalize the sequence
        normalized_seq = seq.upper().replace("T", "U")
        
        # 2. Split into codons
        return [normalized_seq[i:i+3] for i in range(0, len(normalized_seq), 3)]

    def preprocess(self, x):
        # Check the type ONLY ONCE
        if isinstance(x, str):
            # Handle the single string case
            return self._process_one_seq(x)
        else:
            # Handle the list case with a single list comprehension
            return [self._process_one_seq(el) for el in x]


rna_codons_tokenizer = RNA_Codons_Tokenizer(vocab_list=RNA_CODONS_VOCAB, version=_version)

# %%
# run a bunch of tests with randomly generated RNA codon sequences

def test_tokenizer(tokenizer, length=21):
    
    assert length % 3 == 0, "Length must be a multiple of 3 for codons."

    import random

    def random_sample(length):
        # get a string of codons, both lower and upper case
        all_nucleotides = RNA_VOCAB  + [nuc.lower() for nuc in RNA_VOCAB ]
        seq = "".join(random.choices(all_nucleotides, k=length))
        return seq

    for _ in range(100):
        seq = random_sample(length)
        tokens = tokenizer.tokenize(seq)
        decoded = tokenizer.detokenize(tokens)
        assert (orig := seq.upper()) == (final := ''.join(decoded)), f"Failed on sequence: {orig} -> {final}"


if __name__ == "__main__":
    test_tokenizer(tokenizer=rna_codons_tokenizer)