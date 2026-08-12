# MaSIF Data Processing (v4_10m)

This directory contains the data processing scripts for precomputing vertices features using MaSIF (Molecular Surface Interaction Fingerprinting) for the v4_10m dataset (AFDB structures at 10 million scale).

**TL;DR**: After precomputing, each protein has 5 residue-level surface features:
- **n_vertices**: Number of surface vertices per residue (proportional to SASA)
- **si_index**: [mean] Shape index of residue vertices
- **charge**: [sum] Poisson-Boltzmann charges of residue vertices
- **hbond**: [sum] Hydrogen-bond donor/acceptor propensity of residue vertices
- **hydrophobicity**: [mean] Hydrophobicity of residue vertices


**Output Paths by Step:**
- **Step 1**: `$LORE_DATA_ROOT/data/intermediate/masif/v4_10m/hfds_chunked/`
- **Step 2**: `$LORE_DATA_ROOT/data/intermediate/masif/v4_10m/parquet/vertex_struct_joined.parquet/`
- **Step 3**: `$LORE_DATA_ROOT/data/intermediate/masif/v4_10m/parquet/chunks_aggregated/`

## Overview

The MaSIF data processing pipeline for v4_10m consists of three main steps:

1. **Compute Vertex Features**: Extract geometric and chemical features from protein molecular surfaces at the vertex level.
2. **Vertex-Structure Left Join**: Merge vertex features with protein structural data using UniProt IDs.
3. **Aggregate Residue Features**: Aggregate vertex-level features to residue-level features using distance cutoff.

Each step is implemented in a dedicated Python script, with SLURM batch scripts provided for running on HPC clusters.

---

## Step 1: Compute Vertex Features (~2-3 days)

### Purpose

Compute surface features for each vertex on the molecular surface of proteins. These features capture geometric properties (shape index) and chemical properties (charge, hydrogen bonding, hydrophobicity).

### How to Compute

**Requirements:**
- An Apptainer/Docker image carrying MaSIF and its native toolchain (REDUCE, MSMS, APBS,
  PyMesh). Build it from the [MaSIF](https://github.com/LPDI-EPFL/masif) recipe; the run
  below expects it at `$LORE_DATA_ROOT/data/intermediate/masif/masif-with-datasets_latest.sif`.
- The MaSIF source tree at `$LORE_DATA_ROOT/data/intermediate/masif/masif_source`
- Python script: `Step1_compute_masif_vertice_feature.py`

**Execution.** This is a ~2–3 day job over 60 shards and is normally run as a scheduler
array job (one task per shard, `N_CPUS // 2` workers each). The work itself is the
`apptainer exec` below — wrap it in whatever your scheduler wants, passing the shard
index as the last argument:

```bash
VERSION=v4_10m
MASIF=$LORE_DATA_ROOT/data/intermediate/masif
SHARD_INDEX=${SLURM_ARRAY_TASK_ID:-0}      # 0..59

apptainer exec \
    --bind "$PWD":/mnt/masif_scripts \
    --bind "$MASIF/masif_source":/mnt/masif_source \
    --bind "$LORE_DATA_ROOT/data/modality/structure/$VERSION/hfds_chunked":/mnt/afdb_datasets \
    --bind "$MASIF/$VERSION/hfds_chunked":/mnt/masif_outputs \
    "$MASIF/masif-with-datasets_latest.sif" \
    python /mnt/masif_scripts/Step1_compute_masif_vertice_feature.py 60 "$SHARD_INDEX"
```

The `/mnt/masif_*` and `/mnt/afdb_datasets` paths are **container mount points**, not
host paths — `Step1_compute_masif_vertice_feature.py` refers to them internally, so keep
the bind targets exactly as above.

### Processing Workflow

For each protein structure, the script performs the following steps:

1. **Protonation**: Add hydrogen atoms to the protein structure using the REDUCE tool
2. **MSMS Computation**: Compute the molecular surface mesh (vertices, faces, normals) using MSMS
3. **Charge Computation**: Calculate hydrogen bond donor/acceptor properties for vertices
4. **Hydrophobicity Assignment**: Assign hydrophobicity values based on amino acid properties
5. **Mesh Fixing**: Regularize the mesh using PyMesh and recompute normals
6. **APBS Calculation**: Compute electrostatic potential using Poisson-Boltzmann continuum electrostatics
7. **Feature Extraction**: Compile all vertex-level features into a unified array
8. **Save to Disk**: Store results in HuggingFace dataset format (Arrow files)

### Output Annotation

**Path**: `$LORE_DATA_ROOT/data/intermediate/masif/v4_10m/hfds_chunked`

**Format**: HuggingFace datasets (Arrow format), organized in batches of ~1000 entries

Each entry contains:

1. **`uniprot_id`** (string)
   - UniProt identifier consistent with AFDB structure files

2. **`vertices_features`** (array of shape `[N_vertices, 7]`)
   - Column 0-2: **3D coordinates** (x, y, z) of surface vertices
   - Column 3: **Shape index** (-1 to 1)
     - -1: highly concave
     - +1: highly convex
     - Computed from principal curvatures: `si = arctan((k1+k2)/(k1-k2)) * (2/π)`
   - Column 4: **Electrostatic charge** (-1 to 1)
     - Computed from Poisson-Boltzmann continuum electrostatics (APBS)
     - Normalized: values clamped to [-3, 3] then scaled to [-1, 1]
   - Column 5: **Hydrogen bonding** (-1 to 1)
     - -1: optimal position for hydrogen bond acceptor
     - +1: optimal position for hydrogen bond donor
   - Column 6: **Hydrophobicity** (-1 to 1)
     - -1: hydrophilic
     - +1: hydrophobic
     - Normalized by dividing by 4.5

### Key Parameters

- **Mesh resolution**: Defined in `masif_opts['mesh_res']`
- **Feature interpolation**: Method defined in `masif_opts['feature_interpolation']`
- **Batch size**: 1000 entries per output file
- **Electrostatics normalization**: Upper threshold = 3, Lower threshold = -3

---

## Step 2: Vertex-Structure Left Join (~1-2 hours)

### Purpose

Merge the computed vertex features with protein structural data based on UniProt IDs. This creates a unified dataset where each protein has both surface features and structural information.

### How to Compute

**Python script**: `Step2_vertex_struct_left_join.py`

**Requirements:**
- Vertex features from Step 1
- Protein structural data (from `prot_struct` modality)

**Execution:**
```bash
# Standard mode (uses 16 processes by default)
python Step2_vertex_struct_left_join.py

# Specify number of processes
python Step2_vertex_struct_left_join.py --num_proc 32

# Test mode (small subset)
python Step2_vertex_struct_left_join.py --test
```

### Processing Workflow

1. **Load Datasets**: 
   - Load all vertex feature datasets from Step 1
   - Load all protein structure datasets
   - Use multiprocessing for parallel loading

2. **Concatenate**: 
   - Combine all vertex datasets into a single dataset
   - Combine all structure datasets into a single dataset

3. **Create Mapping**: 
   - Build a dictionary mapping UniProt IDs to row indices in the structure dataset
   - Enables fast lookup during merging

4. **Left Join**: 
   - For each protein in the vertex dataset, retrieve its structure from the structure dataset
   - Use dictionary-based lookup for efficiency
   - Proteins without matching structures get `None` values

5. **Save Result**: 
   - Write merged dataset to disk in HuggingFace dataset format
   - Use parallel writing for efficiency

### Input Paths

- **Vertex features**: `data/intermediate/masif/v4_10m/hfds_chunked`
- **Protein structures**: `data/modality/prot_struct/v4_10m/hfds_chunked`

### Output Annotation

**Path**: `$LORE_DATA_ROOT/data/intermediate/masif/v4_10m/parquet/vertex_struct_joined.parquet/`

**Format**: HuggingFace dataset

Each entry contains:

1. **`uniprot_id`** (string)
   - UniProt identifier

2. **`vertices_features`** (array of shape `[N_vertices, 7]`)
   - Same as Step 1 output (see above)

3. **`prot_struct`** (structure data)
   - Protein structural information from the structure modality
   - Contains atom coordinates, residue information, etc.
   - `None` if no matching structure found

### Key Parameters

- **Batch size**: 1024 for map operations
- **Number of processes**: Configurable via `--num_proc` (default: 16)
- **Join key**: `uniprot_id`


---

## Step 3: Aggregate Residue Features (~2 hours)

### Purpose

Aggregate vertex-level surface features to residue-level features. For each residue, identify nearby surface vertices (within 2.8Å) and compute aggregated statistics (mean or sum) of the vertex features.

### How to Compute

**Python script**: `Step3_aggregate_residue_feature.py`

**SLURM script**: `SLURM_Step3_aggregation.sh`

**Requirements:**
- Merged dataset from Step 2 (vertex features + protein structures)
- Biotite library for structure manipulation
- Sufficient memory for distance calculations

**SLURM Configuration:**
- Job array: 0-59 (60 parallel jobs)
- Partition: ccm
- Time limit: 2 hours
- Workers per job: N_CPUS // 2 (parallel processing within each job)

**Execution:**
```bash
sbatch SLURM_Step3_aggregation.sh
```

### Processing Workflow

For each protein in the dataset:

1. **Load Data**:
   - Load vertex features (N_vertices × 7 array)
   - Load protein structure (atom coordinates)

2. **Compute Distances**:
   - Calculate Euclidean distances between all atoms and all vertices
   - Use `scipy.spatial.distance.cdist` for efficient computation

3. **Residue-wise Minimum Distance**:
   - For each residue, find the minimum distance to each vertex
   - Use `biotite.structure.apply_residue_wise` for grouping by residue

4. **Vertex Assignment**:
   - Assign vertices to residues if distance < 2.8Å
   - Create boolean mask: `res_vertice_bool[residue, vertex] = 1` if distance < 2.8Å

5. **Feature Aggregation**:
   - **n_vertices**: Count of surface vertices per residue (sum of boolean mask)
   - **si_index**: Mean shape index of assigned vertices
   - **charge**: Sum of Poisson-Boltzmann charges
   - **hbond**: Sum of hydrogen bond donor/acceptor values
   - **hydrophobicity**: Mean hydrophobicity of assigned vertices

6. **Handle Missing Data**:
   - Residues with no assigned vertices (buried residues) get `NaN` values

7. **Save Results**:
   - Write to Parquet format for efficient storage and fast loading

### Input Path

**Path**: `$LORE_DATA_ROOT/data/intermediate/masif/v4_10m/parquet/vertex_struct_joined.parquet/`

### Output Annotation

**Path**: `$LORE_DATA_ROOT/data/intermediate/masif/v4_10m/parquet/chunks_aggregated/`

**Format**: Parquet files (one per job chunk)

**File naming**: `masif10_aggregated_chunk_{JOB_ID}.parquet`

Each Parquet file contains:

1. **`uniprot_id`** (string)
   - UniProt identifier

2. **`n_vertices`** (array of shape `[N_residues]`)
   - Number of surface vertices within 2.8Å of each residue
   - Proportional to solvent-accessible surface area (SASA)
   - Integer values; `NaN` for buried residues with no surface vertices

3. **`si_index`** (array of shape `[N_residues]`)
   - **[mean]** shape index of vertices assigned to each residue
   - Range: -1 (concave) to +1 (convex)
   - `NaN` for buried residues

4. **`charge`** (array of shape `[N_residues]`)
   - **[sum]** Poisson-Boltzmann electrostatic charges
   - Summed across all vertices within 2.8Å
   - `NaN` for buried residues

5. **`hbond`** (array of shape `[N_residues]`)
   - **[sum]** hydrogen bond donor/acceptor propensity
   - Summed across all vertices within 2.8Å
   - `NaN` for buried residues

6. **`hydrophobicity`** (array of shape `[N_residues]`)
   - **[mean]** hydrophobicity of vertices assigned to each residue
   - Range: -1 (hydrophilic) to +1 (hydrophobic)
   - `NaN` for buried residues

### Key Parameters

- **Distance cutoff**: 2.8Å (vertices within this distance are assigned to a residue)
- **Aggregation methods**:
  - Count: `n_vertices` (sum of boolean mask)
  - Mean: `si_index`, `hydrophobicity` (divided by n_vertices)
  - Sum: `charge`, `hbond` (dot product with boolean mask)
- **Number of jobs**: 60 (dataset split into 60 chunks)
- **Workers per job**: N_CPUS // 2

### Performance Notes

- Processing time per protein: ~0.1-1 seconds (depends on protein size)
- Distance calculation: Most computationally intensive step
- Total runtime per job: ~1-2 hours (for ~100k-200k proteins per job)
- Memory usage: ~10-50 GB per job (depends on protein size distribution)

### Error Handling

- Proteins with missing structure or vertex features are skipped
- Failed proteins are logged with error messages
- Output includes `None` values for failed proteins (can be filtered later)

---

## References

- Gainza, P. et al. (2020). "Deciphering interaction fingerprints from protein molecular surfaces using geometric deep learning." *Nature Methods*, 17(2), 184-192.

- Grudman, S., Fajardo, J. E., & Fiser, A. (2023). Optimal selection of suitable templates in protein interface prediction. *Bioinformatics*, 39(9), btad510.

---

## Troubleshooting

### Common Issues

**Step 1:**
- **Missing APBS/MSMS tools**: Ensure the Apptainer image contains all required dependencies
- **Out of memory errors**: Reduce the number of workers (`N_WORKERS`)
- **Mesh computation failures**: Some structures may fail due to malformed coordinates; these are logged and skipped

**Step 2:**
- **Memory issues during loading**: Reduce `num_proc` or process in smaller batches
- **Mismatched IDs**: Verify that UniProt IDs in vertex and structure datasets are consistent
- **Output path exists**: Remove the existing output file before rerunning

### Monitoring Progress

Check SLURM logs:
```bash
# Step 1 logs
tail -f $LORE_DATA_ROOT/data/intermediate/masif/v4_10m/logs/job_masif_features_*_run.out

# Step 3 logs
tail -f aggregation_logs/masif_aggregation_*.log

# Check for errors in Step 1
grep -i error $LORE_DATA_ROOT/data/intermediate/masif/v4_10m/logs/job_masif_features_*_run.err

# Check for errors in Step 3
grep -i error aggregation_logs/masif_aggregation_*.log
```
