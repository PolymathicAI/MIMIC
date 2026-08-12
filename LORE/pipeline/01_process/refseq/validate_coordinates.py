#!/usr/bin/env python3

"""
Validate coordinate transformations in sequence_tools.py
Checks:
1. The -1 offset in coordinate calculations
2. The +1 in sequence length checks
3. Proper handling of start/stop codons
4. Validation of non-standard start codons and their tags
"""

import sys
sys.path.append('../../../')

import pandas as pd
import numpy as np
import duckdb
from lore import paths
from lore import logger


def is_valid_coordinate(coord):
    """Check if coordinate is valid (not None and not NaN)"""
    if coord is None:
        return False
    if pd.isna(coord):
        return False
    return True


def validate_coordinates(isoform, signal_codons_df, rna_seq_df):
    """Validate coordinate transformations for a single isoform"""
    print(f"\nValidating {isoform['feature_id']} ({isoform['strand']} strand)")
    
    # Get the sequence and signal codons from pre-computed modalities
    genome_feature_id = isoform['genome'] + "_" + isoform['feature_id']
    
    # Get sequence from pre-computed modality
    sequence = rna_seq_df[rna_seq_df['genome_feature_id'] == genome_feature_id]['rna_seq'].iloc[0]
    if sequence is None:
        print("Failed to get sequence from pre-computed modality")
        return None
    
    # Get signal codons from pre-computed modality
    signal_codons = signal_codons_df[signal_codons_df['genome_feature_id'] == genome_feature_id]['signal_codons'].iloc[0]
    
    # Check sequence length
    expected_length = isoform['root_end'] - isoform['root_start'] + 1
    print(f"Sequence length: {len(sequence)} (expected {expected_length})")
    
    # Check signal codons (start and stop)
    if signal_codons is not None:
        # Find positions of signal codons (True values in the boolean array)
        signal_positions = np.where(signal_codons)[0]
        
        for pos in signal_positions:
            codon_seq = sequence[pos:pos+3]
            print(f"Signal codon at position {pos}: {codon_seq}")
            
            # Check if it's a start codon (first position in signal_codons)
            if pos == signal_positions[0]:
                # Check if it's a non-standard start codon
                is_non_standard = codon_seq not in ['ATG']
                has_non_atg_tag = 'non_ATG_start' in isoform.get('tags', [])
                
                if is_non_standard:
                    print(f"Non-standard start codon detected: {codon_seq}")
                    if has_non_atg_tag:
                        print("✓ Has non_ATG_start tag")
                    else:
                        print("✗ Missing non_ATG_start tag")
                elif has_non_atg_tag:
                    print("✗ Has non_ATG_start tag but uses standard ATG start codon")
            
            # Check if it's a stop codon (last position in signal_codons)
            elif pos == signal_positions[-1]:
                if codon_seq not in ['TAA', 'TAG', 'TGA']:
                    print(f"WARNING: Invalid stop codon sequence: {codon_seq}")
        
    return {'rna_seq': sequence, 'signal_codons': signal_codons}


def main():
    # Connect to DuckDB
    con = duckdb.connect(database=':memory:')
    
    # Load data using DuckDB
    parquet_path = str(paths.get_path(data_type="data", stage="intermediate", name="transcripts", version="1.1") / "final_merged_annotations.parquet")
    
    # Create a view of the parquet file
    con.execute(f"CREATE VIEW transcripts AS SELECT * FROM read_parquet('{parquet_path}')")
    
    # Load pre-computed modalities
    signal_codons_path = str(paths.get_path(data_type="data", stage="modality", name="signal_codons", version="1.1", fmt="parquet") / "dataset.parquet")
    rna_seq_path = str(paths.get_path(data_type="data", stage="modality", name="rna_seq", version="1.1", fmt="parquet") / "dataset.parquet")
    
    signal_codons_df = pd.read_parquet(signal_codons_path)
    rna_seq_df = pd.read_parquet(rna_seq_path)
    
    # Get total count of features
    total_count = con.execute("SELECT COUNT(*) FROM transcripts").fetchone()[0]
    
    # Sample 5 random features
    random_sample_size = 5
    random_features = con.execute(f"""
        SELECT * FROM transcripts 
        ORDER BY RANDOM() 
        LIMIT {random_sample_size}
    """).df()
    
    # Sample 5 features with start/stop codons
    codon_sample_size = 5
    codon_features = con.execute(f"""
        SELECT * FROM transcripts 
        WHERE start_codon_coordinates IS NOT NULL 
        AND stop_codon_coordinates IS NOT NULL
        ORDER BY RANDOM() 
        LIMIT {codon_sample_size}
    """).df()
    
    # Combine the samples
    test_df = pd.concat([random_features, codon_features], ignore_index=True)
    
    print(f"Sampled {random_sample_size} random features and {codon_sample_size} features with start/stop codons")
    print(f"Total features in dataset: {total_count}")
    print("\nTest cases:")
    print(test_df[['feature_id', 'strand', 'feature_start', 'feature_end', 'start_codon_coordinates', 'stop_codon_coordinates']].head())
    
    # Test each case
    results = []
    for _, isoform in test_df.iterrows():
        # Check for valid coordinates before validation
        if not is_valid_coordinate(isoform['feature_start']) or not is_valid_coordinate(isoform['feature_end']):
            print(f"\nSkipping {isoform['feature_id']} due to invalid coordinates")
            continue
            
        annotations = validate_coordinates(isoform, signal_codons_df, rna_seq_df)
        if annotations is not None:
            # Get start codon sequence if available
            start_codon_seq = None
            if annotations['signal_codons'] is not None:
                signal_positions = np.where(annotations['signal_codons'])[0]
                if len(signal_positions) > 0:
                    start_codon_seq = annotations['rna_seq'][signal_positions[0]:signal_positions[0]+3]
            
            results.append({
                'feature_id': isoform['feature_id'],
                'strand': isoform['strand'],
                'sequence_length': len(annotations['rna_seq']),
                'expected_length': isoform['root_end'] - isoform['root_start'] + 1,
                'has_signal_codons': annotations['signal_codons'] is not None,
                'start_codon_seq': start_codon_seq,
                'is_non_standard_start': start_codon_seq not in ['ATG'] if start_codon_seq else None,
                'has_non_atg_tag': 'non_ATG_start' in isoform.get('tags', [])
            })
    
    # Print summary
    print("\nSummary:")
    results_df = pd.DataFrame(results)
    print(results_df)
    
    # Check for any issues
    issues = []
    for _, row in results_df.iterrows():
        if row['sequence_length'] != row['expected_length']:
            issues.append(f"Length mismatch for {row['feature_id']}: got {row['sequence_length']}, expected {row['expected_length']}")
        if row['has_signal_codons'] and row['is_non_standard_start'] and not row['has_non_atg_tag']:
            issues.append(f"Non-standard start codon {row['start_codon_seq']} missing non_ATG_start tag for {row['feature_id']}")
        if row['has_signal_codons'] and not row['is_non_standard_start'] and row['has_non_atg_tag']:
            issues.append(f"Standard ATG start codon incorrectly tagged with non_ATG_start for {row['feature_id']}")
    
    if issues:
        print("\nIssues found:")
        for issue in issues:
            print(f"- {issue}")
    else:
        print("\nNo issues found in coordinate transformations")


if __name__ == "__main__":
    main() 