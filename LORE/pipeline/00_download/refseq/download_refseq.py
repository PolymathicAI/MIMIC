#%%
import os
import subprocess
import sys

from lore import logger
from lore import paths

import multiprocessing

import pandas as pd
from tqdm import tqdm

#%% Make the directory for the data


logger.remove()
logger.add(sys.stderr, level="INFO")

path = paths.get_path(data_type="data", stage="downloads", name="refseq", version="229")

# create the directory if it does not exist
os.makedirs(path, exist_ok=True)

logger.debug(f"Directory created at {path} for refseq data")


#%%

# set the working directory to path
logger.debug(f"Working directory set to {path}")
os.chdir(path)


#%% Download NCBI Datasets Download Tool

if not os.path.exists("datasets"):
    logger.debug("Downloading NCBI Datasets Download Tool")
    subprocess.run(["curl", "-o", "datasets", "https://ftp.ncbi.nlm.nih.gov/pub/datasets/command-line/v2/linux-amd64/datasets"], check=True)
    subprocess.run(["chmod", "+x", "datasets"], check=True)
    logger.debug("NCBI Datasets Download Tool downloaded")
else:
    logger.debug("NCBI Datasets Download Tool already exists")


logger.info("NCBI Datasets Download Tool is ready to use. You can run the command './datasets' to see the available options.")
#%%

# Download the assembly summary file
assembly_summary_url = "https://ftp.ncbi.nlm.nih.gov/genomes/ASSEMBLY_REPORTS/assembly_summary_refseq.txt"
assembly_summary_file = "assembly_summary_refseq.txt"

if not os.path.exists(assembly_summary_file):
    logger.debug(f"Downloading assembly summary file from {assembly_summary_url}")
    subprocess.run(["wget", assembly_summary_url, "-O", assembly_summary_file], check=True)
    logger.debug("Assembly summary file downloaded")
else:
    logger.debug("Assembly summary file already exists")

logger.debug(f"Loading assembly summary file {assembly_summary_file}")
df_assembly_summary = pd.read_csv(assembly_summary_file, sep='\t', header=1)
logger.info(f"Assembly summary file loaded with {df_assembly_summary.shape[0]} rows and {df_assembly_summary.shape[1]} columns")

#%%

# Filter the dataset accessions file

# Filter the assembly summary file for reference genomes
# Log the filtering process for reference genomes
logger.debug("Filtering the assembly summary file for reference genomes, annotation provider as 'NCBI RefSeq', and assembly level as 'Complete Genome'")
df_reference_genome = df_assembly_summary[df_assembly_summary['refseq_category'] == 'reference genome']
# df_refseq = df_reference_genome[df_reference_genome['annotation_provider'] == 'NCBI RefSeq']
df_complete = df_reference_genome[df_reference_genome['assembly_level'] == 'Complete Genome']


species_names = [
    "Drosophila melanogaster",
    "Caenorhabditis elegans",
    "Danio rerio",
    "Saccharomyces cerevisiae",
    "Schizosaccharomyces pombe",
    "Arabidopsis thaliana",
    # "Rattus norvegicus", # Updated rat build below to match CAGE
    "Xenopus tropicalis",
    "Dictyostelium discoideum",
    "Apis mellifera",
    "Escherichia coli",
    "Bacillus subtilis",
    "Zea mays",
    "Glycine max"
]

# Filter the assembly summary file for the species names
reference_species = df_reference_genome[df_reference_genome['organism_name'].isin(species_names)]

# Append these rows to the existing DataFrame
df_complete = pd.concat([df_complete, reference_species]).drop_duplicates()

datasets = df_complete['#assembly_accession'].tolist()

# Add specific reference genome accessions that may not be in the filtered sets
additional_accessions = ["GCF_000772875.2", "GCF_000002315.4", "GCF_000002285.3", "GCF_000001895.5"] # Rheseus macaque, Chicken, Dog, Rat
extra_datasets = [acc for acc in additional_accessions if acc not in datasets]
datasets.extend(extra_datasets)

# Log the additional accessions
logger.debug(f"Added {len(extra_datasets)} additional reference genome accessions")

logger.info(f"Filtered assembly summary file for reference genomes with assembly level as 'Complete Genome' or selected reference species with {len(datasets)} entries")


#%%

def download_and_unzip(dataset):
    if not os.path.exists(dataset):

        # Download the dataset if it does not exist
        try:
            logger.debug(f"Downloading the genome {dataset} with NCBI Datasets Download Tool")
            subprocess.run([
            "./datasets",
            "download",
            "genome",
            "accession",
            dataset,
            "--dehydrated",
            "--filename",
            f"{dataset}.zip",
            "--include",
            "gff3,genome,protein,rna"
            ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            logger.debug(f"Download completed for {dataset}")
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to download genome {dataset}: {e}\n")
            return

        logger.debug(f"Unzipping the downloaded file for {dataset}")
        subprocess.run(['unzip', f'{dataset}.zip', '-d', f'{dataset}'], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        logger.debug(f"Unzipping completed for {dataset}")

        logger.debug(f"Removing the zip file for {dataset}")
        os.remove(f"{dataset}.zip")
        logger.debug(f"Zip file removed for {dataset}")

        readme_path = os.path.join(dataset, "README.md")
        if os.path.exists(readme_path):
            logger.debug(f"Removing README.md for {dataset}")
            os.remove(readme_path)
            logger.debug(f"README.md removed for {dataset}")
        else:
            logger.debug(f"README.md does not exist for {dataset}")
    else:
        logger.debug(f"Dataset directory {dataset} already exists. Skipping download.")


    # Rehydrate the dataset
    try:
        logger.debug(f"Rehydrating the dataset for {dataset}")
        subprocess.run([
        "./datasets",
        "rehydrate",
        "--directory",
        f"{dataset}"
        ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        logger.debug(f"Rehydration completed for {dataset}")
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to rehydrate dataset {dataset}: {e}\n")
        return


#%%
# Download and rehydrate the datasets

with multiprocessing.Pool() as pool:
    list(tqdm(pool.imap(download_and_unzip, datasets), total=len(datasets)))

    logger.info("All datasets have been downloaded and rehydrated. Run verify_download.py to verify the download checksums. Rerun this script to download any missing datasets.")

