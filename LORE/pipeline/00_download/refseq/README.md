# RefSeq Data Processing

This directory contains scripts for downloading and verifying RefSeq data.

## Scripts

### `download_refseq.py`
- **Description**: Automates the download of NCBI RefSeq data. Currently configured to download only annotated complete reference genomes. Uses the NCBI datasets CLI tool for download and rehydration. Genomes are downloaded into separate directories. Each directory contains at minimum a sequence (.fna) file and an annotation (.gff) file. RNA and protein sequence files are also downloaded if available, but are not currently used. 

### `verify_download.py`
- **Description**: Verifies the integrity of downloaded RefSeq files using the checksum included in the download. 

