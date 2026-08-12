""" Process CAGE data files to extract cell type, replicate, and sample ID information.

This script reads CAGE data files, extracts relevant metadata from column names,
and saves the structured data into CSV files along with metadata in pickle format.
Note that this is matching on string patterns in the column names, so some minor leakage
may occur if the naming conventions are not strictly followed.

This script is designed to be run from the command line with two arguments:
1. RAW_LOC: The directory where the raw CAGE data files are located.
2. SAVE_LOC: The directory where the processed files should be saved.

Set of organisms for which CAGE data is available if currently fixed
"""

import os
import re
import urllib.parse
import pandas as pd
import pickle
import sys

files_avail = {
    'rn6': 'rn6.cage_peak_tpm.osc.txt', 
    'rheMac8': 'rheMac8.cage_peak_tpm.osc.txt', 
    'galGal5': 'galGal5.cage_peak_tpm.osc.txt', 
    'canFam3': 'canFam3.cage_peak_tpm.osc.txt', 
    'mm9': 'mm9.cage_peak_phase1and2combined_tpm.osc.txt', 
    'hg19': 'hg19.cage_peak_phase1and2combined_tpm.osc.txt'
}

def parse_column_name(col):
    col = urllib.parse.unquote(col)
    if col.startswith("tpm."):
        col = col[4:]
    match = re.match(r"(.+?)\.?(CNhs\d+\.\d+-\w+(?:\.\w+)?$)", col)
    if not match:
        return None
    cell_type_raw, sample_id = match.groups()
    cell_type_raw = cell_type_raw.strip()

    replicate = timepoint = None

    # Existing pattern for donor/pool/day
    match_extra = re.match(r"(.+?)[,\s]+(donor\d+|pool\d+|day\d+)$", cell_type_raw, re.IGNORECASE)
    if match_extra:
        cell_type_raw, qualifier = match_extra.groups()
        qualifier = qualifier.strip().lower()
        if qualifier.startswith("donor") or qualifier.startswith("pool"):
            replicate = qualifier
        elif qualifier.startswith("day"):
            timepoint = qualifier
        if timepoint:
            cell_type_raw = cell_type_raw.strip() + " " + timepoint
    
    if not replicate:
        # Match any known replicate keywords
        repl_patterns = r'\b(biol_rep[l]?\d+|tech_rep\d+|donor\d+|pool\d+|donation\d+)\b'
        fallback_repl = re.search(repl_patterns, cell_type_raw, re.IGNORECASE)
        if fallback_repl:
            replicate = fallback_repl.group(1)
            cell_type_raw = re.sub(re.escape(replicate), '', cell_type_raw, flags=re.IGNORECASE).strip()
            
    return cell_type_raw.strip(), replicate, sample_id


def load_cage_data(species):
    """ Load CAGE data for a given species."""
    file = RAW_LOC+files_avail.get(species)
    if not file:
        raise ValueError(f"No CAGE data available for species: {species}")
    with open(file) as f:
        comment_lines = [line.strip() for line in f if line.startswith('#')]

    # Read the rest (non-comment lines) into a DataFrame
    df = pd.read_csv(
        file,
        sep='\t',
        comment='#'
    )
    stats = df.iloc[:2]
    df = df.iloc[2:]
    df['species'] = species
    df.set_index('00Annotation', inplace=True)
    # rename index name
    df.index.name = None
    
    return df, stats, comment_lines

def process_cell_type(df):
    """ Process cell type into donor, pool, or timepoint."""
    cell_type, replicate, sample_id = [], [], []
    for col in df.columns:
        parsed = parse_column_name(col)
        if parsed:
            cell_type_raw, rep, samp_id = parsed
            cell_type.append(cell_type_raw)
            replicate.append(rep if rep else '')
            sample_id.append(samp_id if samp_id else '')
        else:
            cell_type.append(col)
            replicate.append('')
            sample_id.append('')
    return cell_type, replicate, sample_id

def save_structured_data(df, filename_base, loc='outputs/'):
    metadata = {}
    cell_type, replicate, sample_id = process_cell_type(df)
    current_cts = cell_type
    print('Unique cell types found:', len(set(current_cts)))
    print('Unique donors found:', len(set(replicate)))
    print('Example donors:', replicate[:10])
    print('Unique sample IDs found:', len(set(sample_id)))
    print('Example sample IDs:', sample_id[:10])
    print('Total shape of DataFrame:', df.shape)
    df.columns = [f'ct{idx+1}' for idx in range(len(current_cts))]
    for ct in df.columns:
        metadata[ct] = {
            'cell_type': cell_type.pop(0),
            'replicate': replicate.pop(0),
            'sample_id': sample_id.pop(0)
        }
    os.makedirs(loc, exist_ok=True)
    df.to_csv(f"{loc}{filename_base}.cage.csv", index=True, sep='\t')
    with open(f"{loc}{filename_base}.cage.metadata.pkl", 'wb') as f:
        pickle.dump(metadata, f)
    return df, metadata

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python script.py <RAW_LOC> <SAVE_LOC>")
        sys.exit(1)

    RAW_LOC = sys.argv[1]
    if not RAW_LOC.endswith("/"):
        RAW_LOC += "/"

    SAVE_LOC = sys.argv[2]
    if not SAVE_LOC.endswith("/"):
        SAVE_LOC += "/"

    for species in files_avail.keys():
        print('*' * 15)
        print(f"Loading data for species: {species}")
        df, stats, comment_lines = load_cage_data(species)
        df.drop(columns=['species'], inplace=True)
        df, metadata = save_structured_data(df, species, loc=SAVE_LOC)

