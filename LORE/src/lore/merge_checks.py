import numpy as np
from biotite.sequence import AlphabetError, CodonTable, NucleotideSequence
from tqdm import tqdm

from lore import logger

check_dictionary = {}

FAILURE_ERROR_THRESH = 5  # percentage of failed checks to raise error
CODON_TABLE = CodonTable.load("Standard")


def register_check(name: str):
    """
    Register a check function
    """

    def decorator(func):
        check_dictionary[name] = func
        return func

    return decorator


def run_check(df, test, test_name: str):
    """
    Run a check on the dataframe and print the results
    """
    tqdm.pandas(desc=f"Running {test_name} checks...")
    result = df.progress_apply(lambda row: test(row), axis=1)

    if result.all():
        logger.info(f"All {test_name} checks passed.")
        return 0, 0.0
    else:
        failed = df[~result]
        n_failed = len(failed) if failed is not None else 0
        pct_failed = (n_failed / len(df)) * 100
        if pct_failed < FAILURE_ERROR_THRESH:
            logger.warning(f"{n_failed} {test_name} checks failed ({pct_failed:.2f}%).")
        else:
            logger.error(f"{n_failed} {test_name} checks failed ({pct_failed:.2f}%).")

        return n_failed, pct_failed


def run_check_suite(df, check_list: list[str], check_suite_name: str):
    logger.info(f"Running checks on {len(df)} rows from {check_suite_name} set")

    failed_results = {}
    for test_name in check_list:
        try:
            n_failed, pct_failed = run_check(df, check_dictionary[test_name], test_name)
            failed_results[test_name] = (n_failed, pct_failed)
        except Exception as e:
            logger.error(f"Error running check {test_name}: {(type(e))} {e}")

    logger.info("=" * 20 + "Checks Done" + "=" * 20)
    return failed_results


def mask_rna_seq(rna_seq: str, mask: np.ndarray) -> str:
    """
    Return only the elements of rna_seq where mask is True
    """
    return "".join([c for c, m in zip(rna_seq, mask, strict=False) if m]).upper()


def convert_string_array(string_array: str) -> np.ndarray:
    """
    Convert a string representation of an array to a numpy array
    Example: "11001" -> np.array([1, 1, 0, 0, 1])
    """
    return np.array([int(char) for char in string_array])


@register_check("aa_dssp_length")
def check_aa_dssp_length(row):
    aa_seq, dssp = row["aa_seq"], row["dssp"]
    return len(aa_seq) == len(dssp)


@register_check("aa_sasa_length")
def check_aa_sasa_length(row):
    aa_seq, sasa = row["aa_seq"], row["sasa"]
    return len(aa_seq) == len(sasa)


@register_check("aa_prot_struct_length")
def check_aa_prot_struct_length(row):
    aa_seq, prot_struct = row["aa_seq"], row["prot_struct"]
    return len(aa_seq) == len(prot_struct)


@register_check("rna_seq_lengths")
def check_rna_lengths(row):
    rna_seq, exon, cds = row["rna_seq"], row["splice_regions"], row["cds"]
    return len(rna_seq) == len(exon) == len(cds)


@register_check("rna_seq_length_relations")
def check_rna_length_relations(row):
    rna_seq, exon, cds = row["rna_seq"], row["splice_regions"], row["cds"]
    return len(rna_seq) >= len(exon) >= len(cds)


@register_check("phylop_length")
def check_phylop_length(row):
    phylop_species = [i for i in row.keys() if i.startswith("phylop_")]
    matched = True
    for spec in phylop_species:
        phylop, rna_seq = row[spec], row["rna_seq"]
        matched = matched & (len(phylop) == len(rna_seq))
    return matched


@register_check("cds_multiple_of_three")
def check_multiple_of_three(row):
    cds = row["cds"]
    cds = convert_string_array(cds)
    return sum(cds) % 3 == 0


@register_check("cds_aa_length")
def check_cds_aa_length(row):
    cds, aa_seq = row["cds"], row["aa_seq"]
    cds = convert_string_array(cds)
    return (sum(cds) / 3) == (len(aa_seq) + 1)


@register_check("cds_has_stop_codon")
def check_cds_has_stop_codon(row):
    rna_seq, cds = row["rna_seq"], row["cds"]
    cds = convert_string_array(cds)
    cds_masked = mask_rna_seq(rna_seq, cds)
    return cds_masked[-3:] in ["TAA", "TAG", "TGA"]


@register_check("cds_has_start_codon")
def check_cds_has_start_codon(row):
    rna_seq, cds = row["rna_seq"], row["cds"]
    cds = convert_string_array(cds)
    cds_masked = mask_rna_seq(rna_seq.upper(), cds)
    return cds_masked[:3] in CODON_TABLE.start_codons()


@register_check("cds_translation")
def check_cds_translation(row):
    rna_seq, cds, aa_seq = row["rna_seq"], row["cds"], row["aa_seq"]
    cds = convert_string_array(cds)
    cds_masked = NucleotideSequence(mask_rna_seq(rna_seq, cds))
    try:
        cds_translated = cds_masked.translate()[0]
    except AlphabetError:
        return False

    if len(cds_translated) == 0:
        return False
    else:
        cds_translated = str(cds_translated[0]).rstrip("*")
        return cds_translated == aa_seq


@register_check("rna_codons_valid")
def check_rna_codons_valid(row):
    rna_codons = row["rna_codons"]
    rna_codon_set = set(i.decode().upper() for i in rna_codons)
    possible_codon_set = set(CODON_TABLE.codon_dict().keys()).union(
        set(CODON_TABLE.start_codons())
    )
    return rna_codon_set.issubset(possible_codon_set)


@register_check("rna_codons_translation")
def check_rna_codons_translation(row):
    rna_codons, aa_seq = row["rna_codons"], row["aa_seq"]
    rna_seq = b"".join(rna_codons).decode().upper()
    rna_sequence = NucleotideSequence(rna_seq)

    try:
        translated_seq = str(rna_sequence.translate(complete=True)).rstrip("*")
    except AlphabetError:
        return False

    return translated_seq == aa_seq
