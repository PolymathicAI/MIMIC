# Library of helper functions for accessing processed RefSeq data

import math
import re

import gffutils
import h5py
import numpy as np
import pandas as pd

from lore import logger
from lore import paths


def load_gffutils_db(genome):
    """
    Load the gffutils database for a given genome.
    Args:
        genome (str): Genome identifier (e.g., 'GCF_000001405.40').
    Returns:
        gffutils.FeatureDB: Loaded gffutils database.
    """
    db_path = (
        paths.get_path(
            data_type="data", stage="intermediate", name="refseq", version="229"
        )
        / genome
        / "annotations.db"
    )

    try:
        logger.debug(f"Loading existing gffutils database from {db_path}")
        return gffutils.FeatureDB(db_path, keep_order=True)
    except Exception as e:
        logger.error(f"Failed to load existing database: {e}")
        return None


def reverse_complement(seq):
    complement = str.maketrans("ACGTacgt", "TGCAtgca")
    return seq.translate(complement)[::-1]


def get_hdf5_file(isoform):
    """
    Get the HDF5 file for a given isoform.
    Args:
        isoform (DataFrame Row): Isoform row from the annotated genome dataframe. Must have 'genome' key.
    Returns:
        str: Path to the HDF5 file.
    """

    # Determine the data source based on the organism name
    if "GCF" in isoform["genome"]:
        data_source = (
            "gencode"
            if isoform["organism_name"] in ["Homo sapiens", "Mus musculus"]
            else "refseq"
        )
        version = "229"
    else:
        data_source = "refseq"
        version = "rna2d_rasp"

    if data_source == "refseq":
        return (
            paths.get_path(
                data_type="data", stage="intermediate", name="refseq", version=version
            )
            / isoform["genome"]
            / "genome_sequences.hdf5"
        )
    elif data_source == "gencode":
        return paths.get_path(
            data_type="data", stage="intermediate", name="gencode"
        ) / (
            ("human" if isoform["organism_name"] == "Homo sapiens" else "mouse")
            + "/genome_sequences.hdf5"
        )
    else:
        raise ValueError(
            f"Unknown data source: {data_source}. Supported sources are 'refseq' and 'gencode'."
        )


def get_feature_coords(isoform, padding=0, precomputed_flanking=True):

    if precomputed_flanking and 'flanking_start' in isoform and pd.notna(isoform['flanking_start']):
        feature_start = isoform['flanking_start']
        feature_end = isoform['flanking_end']
    else:
        feature_start = isoform["root_start"]
        feature_end = isoform["root_end"]

    feature_start = feature_start - padding - 1
    feature_end = feature_end + padding

    if feature_start < 0:
        feature_start = 0
        logger.debug(
            f"Start position for genome {isoform['genome']}, {isoform['feature_id']} is negative. Setting to 0."
        )

    return [feature_start, feature_end]


def get_sequence_from_annotation(isoform, padding=0, hdf=None, drop_N=True, drop_N_count=800, drop_N_pct=0.4, dry=False, precomputed_flanking=True):
    """
    Retrieve a specific region of the genome from an HDF5 file.
    Args:
        isoform (DataFrame Row): Isoform row from the annotated genome dataframe. Must have 'genome', 'seqid', 'start', and 'end' keys.
        padding (int): Number of bases to pad the start and end positions.
        precomputed_flanking (bool): Whether to use flanking coordinates if available (default: True).
    Returns:
        str: The sequence of the specified region.
    """

    chrom = isoform["seqid"]

    # Get the start and end positions based of the top level feature
    start, end = get_feature_coords(isoform, padding, precomputed_flanking=precomputed_flanking)

    if dry:
        return start

    if hdf is None:
        # Get the HDF5 file path if loading a single isoform
        with h5py.File(get_hdf5_file(isoform), "r") as hdf_fi:
            seq = b"".join(hdf_fi[chrom]["sequence"][start:end]).decode()
    else:
        # Else if loading in batch, hdf file handle will already be open and passed in
        seq = b"".join(hdf[chrom]["sequence"][start:end]).decode()

    # Consider what we want to do in this case. Should we just clip the annotation to the end here?
    # Also, figure out why this is happening so often
    if len(seq) < isoform["root_end"] - isoform["root_start"] + 1:
        logger.warning(
            f"Sequence length for genome {isoform['genome']}, feature_id {isoform['feature_id']} is less than expected. Length: {len(seq)}"
        )
        return None

    N_count = seq.count("N") + seq.count("n")

    if drop_N and (N_count > drop_N_count or N_count / len(seq) > drop_N_pct):
        logger.warning(
            f"Sequence for genome {isoform['genome']}, feature_id {isoform['feature_id']} contains too many 'N' bases. Skipping."
        )
        return None

    if isoform["strand"] == "-":
        seq = reverse_complement(seq)

    # Check if the sequence length matches the expected length
    # if len(seq) != end - start:
    #     logger.warning(f"Sequence length mismatch for genome {isoform['genome']}, feature_id {isoform['feature_id']}: {len(seq)} != {end - start}. Padding out of bounds at the end of the chromosome assembly.")

    # return np.array(list(seq), dtype='U1')
    return seq


def logical_or_string(str1, str2):
    """
    Perform logical OR operation on two strings of '0's and '1's.
    Returns a new string where each position is '1' if either input has '1' at that position.
    """
    arr1 = np.frombuffer(str1.encode("ascii"), dtype=np.uint8)
    arr2 = np.frombuffer(str2.encode("ascii"), dtype=np.uint8)

    # The ASCII codes for '0' and '1' differ only in the LSB.
    # Mask with 1 to turn b'0'/b'1' → 0/1, then OR.
    combined_arr = (arr1 & 1) | (arr2 & 1)  # dtype uint8, values 0/1

    # Convert back to a string of '0's and '1's
    return "".join(combined_arr.astype(str))


def create_junction_mask(region_mask):
    """
    Create a junction mask from a region mask.
    Marks:
    - 0 for non-junction
    - 1 for acceptor (last BP of intron before exon)
    - 2 for donor (first BP of intron after exon)
    - 3 for isoform start (first acceptor)
    - 4 for isoform end (last donor)

    Args:
        region_mask (str): String of '0's and '1's indicating regions

    Returns:
        str: String of '0's, '1's, '2's, '3's, and '4's indicating junctions
    """
    mask = np.array(list(region_mask), dtype=int)
    junction_mask = np.zeros_like(mask)

    # Find acceptor junctions (0->1 transitions), subtract 1 to mark last base before region
    acceptor_indices = np.where(np.diff(np.concatenate(([0], mask, [0]))) == 1)[0] - 1
    acceptor_indices = acceptor_indices[acceptor_indices >= 0]
    junction_mask[acceptor_indices] = 1

    # Find donor junctions (1->0 transitions), these mark first base after region
    donor_indices = np.where(np.diff(np.concatenate(([0], mask, [0]))) == -1)[0]
    donor_indices = donor_indices[donor_indices < len(mask)]
    junction_mask[donor_indices] = 2

    # Mark first acceptor as isoform start (3)
    if len(acceptor_indices) > 0:
        junction_mask[acceptor_indices[0]] = 3

    # Mark last donor as isoform end (4)
    if len(donor_indices) > 0:
        junction_mask[donor_indices[-1]] = 4

    return ''.join(junction_mask.astype(str))


def missing_none_or_na(val) -> bool:
    # If it's array-like, don't treat it as missing
    if isinstance(val, np.ndarray | list | tuple | pd.Series):
        return False
    return pd.isna(val)  # True for None, <NA>, NaN, NaT

def get_nucleotide_annotations(isoform, padding=0, hdf=None, precomputed_flanking=True, single_exon_junctions=True, drop_N=True):
    # Get the sequence
    sequence = get_sequence_from_annotation(isoform, padding, hdf, precomputed_flanking=precomputed_flanking, drop_N=drop_N)

    if sequence is None or isoform["splice_region_coordinates"] is None:
        return None

    # Initialize mask to all False
    annotations = {}

    # Get the start position (accounting for padding)
    start_pos = get_feature_coords(isoform, padding, precomputed_flanking=precomputed_flanking)[0]

    # List of features to look for
    features = [
        "CDS",
        "splice_region",
        "five_prime_UTR",
        "three_prime_UTR",
        "start_codon",
        "stop_codon",
    ]

    # For each feature type, check if it exists and create a mask if it does
    for feature in features:
        col = f"{feature}_coordinates"
        coords = isoform.get(col, None)

        if feature == "splice_region":
            feature = "splice_regions"

        if missing_none_or_na(coords):

            # If the feature is not present in the isoform, set it to None
            annotations[feature] = None
            continue

        if feature == "CDS" and isoform["feature_type_clean"] == "pseudogene":
            # If the feature is a pseudogene, we don't want to create a mask for CDS
            annotations[feature] = None
            continue

        # Initialize mask for this feature
        annotations[feature] = "0" * len(sequence)

        try:
            for coord_pair in coords:
                feat_start, feat_end = coord_pair
                # Convert coordinates to positions in the padded sequence
                rel_start = (
                    (feat_start - 1) - start_pos
                )  # Convert feat_start to 0-based, then to relative position
                rel_end = feat_end - start_pos

                # Make sure coordinates are within bounds
                if rel_start < 0:
                    logger.debug(
                        f"Feature coordinates for {isoform['feature_id']} {feature} in genome {isoform['genome']} extend before sequence start. Clipping to 0."
                    )
                    rel_start = 0
                if rel_end > len(sequence):
                    logger.debug(
                        f"Feature coordinates for {isoform['feature_id']} {feature} in genome {isoform['genome']} extend beyond sequence end. Clipping to {len(sequence)}."
                    )
                    rel_end = len(sequence)

                if rel_start > rel_end:
                    logger.warning(
                        f"Feature coordinates for {isoform['feature_id']} {feature} in genome {isoform['genome']} are invalid: {rel_start} > {rel_end}"
                    )
                    continue

                if rel_start < len(sequence) and rel_end > 0:
                    # Update the relevant portion to 1s
                    annotations[feature] = (
                        annotations[feature][:rel_start]
                        + "1" * (rel_end - rel_start)
                        + annotations[feature][rel_end:]
                    )
        except Exception as e:
            logger.error(
                f"Error processing coordinates for {feature} for isoform: {isoform}"
            )
            logger.error(f"Error: {e}")
            raise e

        # Flip the mask if the isoform is on the negative strand
        if isoform["strand"] == "-":
            annotations[feature] = annotations[feature][::-1]

    # Make Splice Junction Modality
    # This mimics SOTA Splicing Models, where instead of marking the entire splice region,
    # we only mark the junctions (the first and last base of each splice region)
    # Mark 0 for non-junction, 1 for acceptor, 2 for donor

    # Only create splice/CDS junctions if there are multiple splice regions
    if annotations['splice_regions'] is not None and (single_exon_junctions or len(isoform['splice_region_coordinates']) > 1):
        annotations['splice_junctions'] = create_junction_mask(annotations['splice_regions'])
    else:
        annotations["splice_junctions"] = None

    # Do the same for CDS Junctions
    if annotations['CDS'] is not None and (single_exon_junctions or len(isoform['CDS_coordinates']) > 1):
        annotations['CDS_junctions'] = create_junction_mask(annotations['CDS'])
    else:
        annotations["CDS_junctions"] = None

    # Group related features

    # Group UTR features
    utr_features = ["five_prime_UTR", "three_prime_UTR"]
    if all(annotations[utr] is not None for utr in utr_features):
        annotations["utr"] = logical_or_string(
            annotations["five_prime_UTR"], annotations["three_prime_UTR"]
        )
    else:
        # If any UTR feature is missing, set combined UTR to None
        annotations["utr"] = None

    # Drop individual UTR features after combining
    for utr in utr_features:
        annotations.pop(utr)

    # Group codon features
    codon_features = ["start_codon", "stop_codon"]
    if all(annotations[codon] is not None for codon in codon_features):
        annotations["signal_codons"] = logical_or_string(
            annotations["start_codon"], annotations["stop_codon"]
        )
    else:
        # If any codon feature is missing, set combined signal_codons to None
        annotations["signal_codons"] = None

    # Drop individual codon features after combining
    for codon in codon_features:
        annotations.pop(codon)

    # Add the sequence to the annotations dictionary
    annotations["rna_seq"] = sequence

    # Return all annotations as a Series
    return pd.Series(annotations)


def get_isoforms_by_root_id(isoform_df, root_id):
    # Get the isoforms for a given gene ID

    isoforms = isoform_df[isoform_df["root_id"] == root_id]

    return isoforms


def get_alternate_isoforms(isoform_df, isoform):
    # Get the alternate isoforms for a given isoform

    isoforms = isoform_df[isoform_df["root_id"] == isoform["root_id"]]

    return isoforms


def get_exon_one_hot(gene, flanking_size=0):
    """Create one-hot encoded exon array for a pyensembl gene object.

    This function generates a binary array marking exonic regions across the gene's
    genomic span. Overlapping exons from different transcripts are combined such that
    each position can have a value > 1 if multiple isoforms include that position.

    Args:
        gene: A pyensembl.Gene object containing gene annotation information
        flanking_size (int): Number of zero-padded positions to add at both ends.
                            Defaults to 0.

    Returns:
        np.ndarray: Integer array where each position indicates the number of
                   transcripts with an exon at that genomic position. Array length
                   is gene_length + 2*flanking_size.
    """
    # Get exons for the gene
    exons = gene.exons

    # Get the gene start and end positions
    gene_start = gene.start
    gene_end = gene.end
    gene_length = gene_end - gene_start + 1

    # Initialize a zero array for the gene length
    exon_one_hot = np.zeros(gene_length, dtype=int)

    # Mark exon regions with 1
    for exon in exons:
        exon_start_relative = exon.start - gene_start
        exon_end_relative = exon.end - gene_start
        exon_one_hot[exon_start_relative:exon_end_relative + 1] += 1

    # Add flanking zeros at the beginning and end
    if flanking_size > 0:
        exon_one_hot = np.concatenate([
            np.zeros(flanking_size, dtype=int),
            exon_one_hot,
            np.zeros(flanking_size, dtype=int)
        ])

    return exon_one_hot


def get_isoform_splicing(gene, flanking_size=0):
    """Create individual isoform splicing patterns for a pyensembl gene object.

    This function extracts splicing information for each transcript isoform of a gene,
    generating separate binary exon masks and junction masks. Handles strand orientation
    by reversing arrays for genes on the minus strand.

    Args:
        gene: A pyensembl.Gene object containing gene annotation information
        flanking_size (int): Number of zero-padded positions to add at both ends.
                            Defaults to 0.

    Returns:
        tuple: A 4-tuple containing:
            - isoforms (list[np.ndarray]): Binary exon masks for each transcript
            - junctions (list[str]): Junction masks for each transcript
            - biotypes (list[str]): Transcript biotype annotations
            - coding (list[bool]): Whether each transcript is protein coding
    """
    gene_start = gene.start
    gene_end = gene.end
    gene_length = gene_end - gene_start + 1

    isoforms = []
    junctions = []
    biotypes = []
    coding = []

    # Get all transcripts for this gene
    for transcript in gene.transcripts:
        exon_one_hot = np.zeros(gene_length, dtype=int)

        # Mark exon regions for this specific transcript
        for exon in transcript.exons:
            exon_start_relative = exon.start - gene_start
            exon_end_relative = exon.end - gene_start
            exon_one_hot[exon_start_relative:exon_end_relative + 1] = 1

        # Add flanking zeros if needed
        if flanking_size > 0:
            exon_one_hot = np.concatenate([
                np.zeros(flanking_size, dtype=int),
                exon_one_hot,
                np.zeros(flanking_size, dtype=int)
            ])

        if gene.strand == '-':
            exon_one_hot = exon_one_hot[::-1]


        isoforms.append(exon_one_hot)
        junctions.append(create_junction_mask(exon_one_hot))
        biotypes.append(transcript.biotype)
        coding.append(transcript.biotype == 'protein_coding')

    # Sort by transcript ID alphabetically
    sorted_indices = sorted(range(len(gene.transcripts)), key=lambda i: gene.transcripts[i].id)

    isoforms = [isoforms[i] for i in sorted_indices]
    junctions = [junctions[i] for i in sorted_indices]
    biotypes = [biotypes[i] for i in sorted_indices]
    coding = [coding[i] for i in sorted_indices]

    return isoforms, junctions, biotypes, coding


def get_clean_feature_type(feature):
    return (
        get_clean_feature_gencode(feature)
        if feature["organism_name"] in ["Homo sapiens", "Mus musculus"]
        else get_clean_feature_refseq(feature)
    )


def get_clean_feature_gencode(feature):
    """
    Determine a simplified feature type based on the transcript_type field in GENCODE
    """

    transcript_type = feature["feature_type"]

    # Check for pseudogene first
    if "pseudo" in transcript_type.lower():
        return "pseudogene"

    # Check for protein coding
    elif "protein_coding" in feature["root_type"]:
        return transcript_type

    elif "gene" in transcript_type:
        # If it contains 'gene' but not 'protein_coding', classify as 'gene'
        return "gene"

    # Check for various RNA types
    elif "rna" in transcript_type.lower():
        # Determine specific RNA type
        if "lnc" in transcript_type.lower():
            return "lncRNA"
        elif "rRNA" in transcript_type:
            return "rRNA"
        elif "tRNA" in transcript_type:
            return "tRNA"
        else:
            return "miscRNA"

    # Check for TEC (To be Experimentally Confirmed)
    elif "TEC" in transcript_type:
        return "TEC"

    # Everything else
    else:
        return "other"


def get_clean_feature_refseq(feature):
    """
    Determine a simplified feature type based on the root_biotype and feature_type fields in RefSeq
    """

    biotype = feature["root_biotype"]
    feature_type = feature["feature_type"]

    # Handle null biotype case
    if pd.isna(biotype):
        if "rna" in feature_type.lower():
            # Normalize lnc_RNA to lncRNA
            if "lnc" in feature_type.lower():
                return "lncRNA"
            return feature_type
        return "trans_splicing_suspected"

    # For protein_coding root biotype
    elif biotype == "protein_coding":
        if feature_type in ["gene", "mRNA"]:
            return "protein_coding"

    # For pseudogene and similar
    elif "pseudo" in biotype.lower():
        return "pseudogene"

    # For tRNA
    elif biotype == "tRNA":
        if feature_type == "tRNA":
            return "tRNA"
        else:
            return "miscRNA"  #'other_tRNA'

    # For rRNA
    elif biotype == "rRNA":
        if feature_type == "rRNA":
            return "rRNA"
        else:
            return "miscRNA"  #'other_rRNA'

    # For lncRNA
    elif biotype == "lncRNA":
        if feature_type == "lnc_RNA":
            return "lncRNA"
        else:
            return "miscRNA"  #'other_lncRNA'

    elif biotype == "transcript" and feature["gbkey"] == "miscRNA":
        return "miscRNA"

    # Handle other biotypes based on name patterns
    elif "rna" in biotype.lower():
        return "miscRNA"
    elif "V_segment" in biotype or "C_region" in biotype:
        return "protein_coding"
    elif "rna" in feature["gbkey"].lower():
        return "miscRNA"
    elif biotype == "segment":
        return "trans_splicing_suspected"
    else:
        return "other"


def splice_sequence(seq, splice_regions):
    """
    Splice a sequence based on splice regions.
    Args:
        seq (str): The original sequence.
        splice_regions (str): A string of '0's and '1's indicating splice regions or CDS.
    Returns:
        str: The spliced sequence.
    """

    # Convert splice mask to boolean array efficiently
    splice_mask = np.frombuffer(splice_regions.encode(), dtype="S1") == b"1"

    # Only keep positions where splice_mask is True
    spliced_seq = "".join(np.array(list(seq))[splice_mask])

    return spliced_seq


def get_rna_codons_from_processed_modalities(row):
    """
    Get RNA codons for a given processed modality row.
    Args:
        row (df row): Row containing processed modality data.
    Returns:
        array: Coding RNA sequence. Each element of the array is a codon (3 nucleotides).
    """

    if (
        "rna_seq" not in row
        or "CDS" not in row
        or row["rna_seq"] is None
        or row["CDS"] is None
    ):
        logger.debug(
            f"RNA sequence or CDS are None for {row['genome_feature_id']}. Skipping RNA codon modality."
        )
        return None

    if row["feature_type"] != "protein_coding":
        logger.debug(
            f"Feature {row['genome_feature_id']} is not protein_coding. Skipping RNA codon modality."
        )
        return None

    cds = row["CDS"]
    rna_seq = row["rna_seq"]
    coding_seq = splice_sequence(rna_seq, cds)

    if len(coding_seq) % 3 != 0:
        logger.debug(
            f"Coding RNA sequence length for {row['genome_feature_id']} is not a multiple of 3. Length: {len(coding_seq)}"
        )
        return None

    return coding_seq


def parse_gc_prt(path):
    """
    Parse the gc.prt file to extract codon tables.
    Args:
        path (str): Path to the gc.prt file.
    Returns:
        dict: Dictionary of codon tables keyed by table ID.
    """

    with open(path) as f:
        txt = f.read()
    blocks = re.findall(
        r'\{[^{}]*?ncbieaa\s+"[A-Z*]+"\s*,[^{}]*?-- Base3\s+[ACGT]+', txt, flags=re.S
    )
    tables = {}
    for b in blocks:
        tid = int(re.search(r"\bid\s+(\d+)\s*,", b).group(1))
        aastr = re.search(r'ncbieaa\s+"([A-Z*]+)"', b).group(1)
        starts = re.search(r'sncbieaa\s+"([-A-Z*]+)"', b).group(1)
        b1 = re.search(r"-- Base1\s+([ACGT]+)", b).group(1)
        b2 = re.search(r"-- Base2\s+([ACGT]+)", b).group(1)
        b3 = re.search(r"-- Base3\s+([ACGT]+)", b).group(1)
        codons = [a + b + c for a, b, c in zip(b1, b2, b3)]
        dna_map = dict(zip(codons, aastr))
        dna_starts = [c for c, s in zip(codons, starts) if s == "M"]
        rna_map = {k.replace("T", "U"): v for k, v in dna_map.items()}
        rna_starts = [c.replace("T", "U") for c in dna_starts]
        tables[tid] = {
            "dna_codon_to_aa": dna_map,
            "dna_start_codons": dna_starts,
            "rna_codon_to_aa": rna_map,
            "rna_start_codons": rna_starts,
        }
    return tables


def get_codon_table(genetic_code):
    """
    Retrieve the codon table for a given genetic code ID.
    Args:
        genetic_code (int): Genetic code ID.
    Returns:
        dict: Codon table with keys "dna_codon_to_aa", "dna_start_codons", "rna_codon_to_aa", and "rna_start_codons".
    """

    codon_tables = parse_gc_prt(
        paths.get_path(data_type="data", stage="downloads", name="refseq") / "gc.prt"
    )

    if genetic_code not in codon_tables:
        raise ValueError(f"Genetic code {genetic_code} not found in codon tables.")

    return codon_tables[genetic_code]


def translate_dna(seq, table):
    """
    Translate a DNA sequence using the provided codon table.
    Uses start codon mapping for the first codon.
    Args:
        seq (str): The DNA sequence to translate.
        table (dict): Codon table with keys "dna_codon_to_aa" and "dna_start_codons".
    Returns:
        str: The translated protein sequence.
    """

    protein = []
    for i in range(0, len(seq), 3):
        codon = seq[i : i + 3]
        if len(codon) < 3:
            break
        if i == 0 and codon in table.get("dna_start_codons", []):
            # Use 'M' for start codon if present in start codons
            protein.append("M")
        else:
            amino_acid = table["dna_codon_to_aa"].get(codon, "-")
            protein.append(amino_acid)
    return "".join(protein)
