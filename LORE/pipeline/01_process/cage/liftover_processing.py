""" LiftOver CAGE annotations from one genome assembly to another.

This script replaces the peak locations from one build to another using a pre-generated liftOver file. 
Metadata is not affected, only .csv is updated with new peak locations.
It is assumed that the liftOver has been performed beforehand and the lifted BED file is available.
Ex. liftOver commands:
./liftOver peaks_mm9.bed mm9ToMm39.over.chain.gz peaks_mm39.bed peaks_unmapped_mm39.bed
./liftOver cage/peaks_hg19.bed hg19ToHg38.over.chain.gz peaks_hg38.bed peaks_unmapped.bed

Usage:
  python lift_cage_annotations.py <cage.csv> <lifted.bed> <output.csv>
"""
import pandas as pd
import sys


def lift_and_save(cage_path, bed_path, output_path):
    print(f"\nProcessing:\n - CAGE: {cage_path}\n - BED: {bed_path}\n - Output: {output_path}")

    # Load CAGE data
    cage = pd.read_csv(cage_path, sep='\t', index_col=0)
    cage['old_annotation'] = cage.index.values
    cage.reset_index(drop=True, inplace=True)

    # Load lifted BED file
    lifted = pd.read_csv(bed_path, sep="\t", header=None,
                         names=["chrom", "start", "end", "old_annotation", "score", "strand"])
    lifted["new_annotation"] = (
        lifted["chrom"] + ":" +
        lifted["start"].astype(str) + ".." +
        lifted["end"].astype(str) + "," +
        lifted["strand"]
    ).astype(str)
    lifted['old_annotation'] = lifted['old_annotation'].astype(str)

    print(lifted.head())
    print(f"Total mapped peaks: {len(lifted)}")

    # Merge on old_annotation
    merged = cage.merge(lifted[["new_annotation", "old_annotation"]],
                        on='old_annotation', how='inner')

    print(merged[['new_annotation', 'old_annotation']].head(10))

    # Final formatting
    merged.index = merged['new_annotation'].values
    merged.drop(columns=['old_annotation', 'new_annotation'], inplace=True)

    # Save result
    merged.to_csv(output_path, sep='\t', index=True)
    print(f"Saved lifted file to: {output_path}")

def main():
    if len(sys.argv) != 4:
        print("Usage:\n  python lift_cage_annotations.py <cage.csv> <lifted.bed> <output.csv>")
        sys.exit(1)

    cage_path = sys.argv[1]
    bed_path = sys.argv[2]
    output_path = sys.argv[3]
    lift_and_save(cage_path, bed_path, output_path)

if __name__ == "__main__":
    main()
