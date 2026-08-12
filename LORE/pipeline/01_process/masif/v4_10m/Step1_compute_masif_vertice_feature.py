#!/usr/bin/python
import numpy as np
import os, sys, glob, shutil
from tqdm import tqdm
import pymesh
from concurrent.futures import ProcessPoolExecutor

masif_source="/mnt/masif_source"
masif_scripts="/mnt/masif_scripts"
sys.path.append(masif_scripts)
sys.path.append(masif_source)

# masif_source=$masif_root/source
# export PYTHONPATH=$PYTHONPATH:$masif_source
# Local includes
from numpy_to_pdb import numpy_to_pdb
from pyarrow_read_hfds import read_multiple_arrow_files

from default_config.masif_opts import masif_opts
from triangulation.computeMSMS import computeMSMS
from triangulation.fixmesh import fix_mesh
from input_output.protonate import protonate
from triangulation.computeHydrophobicity import computeHydrophobicity
from triangulation.computeCharges import computeCharges, assignChargesToNewMesh
from triangulation.computeAPBS import computeAPBS
from triangulation.compute_normal import compute_normal

from datasets import Dataset,concatenate_datasets

def normalize_electrostatics(in_elec):
    """
        Normalize electrostatics to a value between -1 and 1
    """
    elec = np.copy(in_elec)
    upper_threshold = 3
    lower_threshold = -3
    elec[elec > upper_threshold] = upper_threshold
    elec[elec < lower_threshold] = lower_threshold
    elec = elec - lower_threshold
    elec = elec / (upper_threshold - lower_threshold)
    elec = 2 * elec - 1
    return elec


def create_mesh(
    vertices,
    faces=[],
    normals=None,
    charges=None,
    hbond=None,
    hphob=None,
    normalize_charges=False,
):
    mesh = pymesh.form_mesh(vertices, faces)
    if normals is not None:
        n1 = normals[:, 0]
        n2 = normals[:, 1]
        n3 = normals[:, 2]
        mesh.add_attribute("vertex_nx")
        mesh.set_attribute("vertex_nx", n1)
        mesh.add_attribute("vertex_ny")
        mesh.set_attribute("vertex_ny", n2)
        mesh.add_attribute("vertex_nz")
        mesh.set_attribute("vertex_nz", n3)
    if charges is not None:
        mesh.add_attribute("vertex_charge")
        if normalize_charges:
            charges = charges / 10
        mesh.set_attribute("vertex_charge", charges)
    if hbond is not None:
        mesh.add_attribute("vertex_hbond")
        mesh.set_attribute("vertex_hbond", hbond)
    if hphob is not None:
        mesh.add_attribute("vertex_hphob")
        mesh.set_attribute("vertex_hphob", hphob)
    return mesh


def compute_features(mesh):
    """
    Compute the features for the mesh vertices, dropped the ddc feature beacuse it is patch based.
    
    Return:
    [N, 7] array of features.
    X, Y, Z, shape index (-1, to 1), charge (-1, to 1), hbonds (-1 to 1), hphob (-1 to 1)
    """
    # Compute the principal curvature components for the shape index. 
    mesh.add_attribute("vertex_mean_curvature")
    H = mesh.get_attribute("vertex_mean_curvature")
    mesh.add_attribute("vertex_gaussian_curvature")
    K = mesh.get_attribute("vertex_gaussian_curvature")
    elem = np.square(H) - K
    # In some cases this equation is less than zero, likely due to the method that computes the mean and gaussian curvature.
    # set to an epsilon.
    elem[elem<0] = 1e-8
    k1 = H + np.sqrt(elem)
    k2 = H - np.sqrt(elem)
    # Compute the shape index 
    si = (k1+k2)/(k1-k2)
    si = np.arctan(si)*(2/np.pi)

    # Normalize the charge, between -1 and 1
    charge = mesh.get_attribute("vertex_charge")
    charge = normalize_electrostatics(charge)
    # Hbond features
    hbond = mesh.get_attribute("vertex_hbond")

    # Hydropathy features
    # Normalize hydropathy by dividing by 4.5
    hphob = mesh.get_attribute("vertex_hphob")/4.5
    features = np.concatenate([mesh.vertices, si.reshape(-1,1), charge.reshape(-1,1), hbond.reshape(-1,1), hphob.reshape(-1,1)], axis=1)
    return features.astype(np.float32)


def process_entry(uniprot_id, atom_array):
    try:
        filename = uniprot_id
        TMP_FOLDER = f"/tmp/masif/{filename}" # Use /tmp file for fast io
        if not os.path.exists(TMP_FOLDER):
            os.makedirs(TMP_FOLDER)
        # Step1: protonate the protein with reduce
        file_base = TMP_FOLDER + "/" + filename
        tmp_path = file_base + ".pdb"
        tmp_protonated_base = TMP_FOLDER + "/reduce_" + filename
        _  = numpy_to_pdb(atom_array, tmp_path) # Convert numpy array to PDB file
        protonate(tmp_path, tmp_protonated_base+".pdb")

        # Step2: commpute MSMS
        vertices1, faces1, normals1, names1, areas1 = computeMSMS(tmp_protonated_base+".pdb", protonate=True)

        # Step3: compute "charged" vertices and 
        vertex_hbond = computeCharges(tmp_protonated_base, vertices1, names1)
        
        # Step4: For each surface residue, assign the hydrophobicity of its amino acid. 
        vertex_hphobicity = computeHydrophobicity(names1)

        # Step5: fix the mesh with pymesh and re-calculate the normals
        # If protonate = false, recompute MSMS of surface, but without hydrogens (set radius of hydrogens to 0).
        vertices2 = vertices1
        faces2 = faces1
        mesh = pymesh.form_mesh(vertices2, faces2)
        regular_mesh = fix_mesh(mesh, masif_opts['mesh_res'])
        vertex_normal = compute_normal(regular_mesh.vertices, regular_mesh.faces)
        # Assign charges on new vertices based on charges of old vertices (nearest neighbor)
        vertex_hbond = assignChargesToNewMesh(regular_mesh.vertices, vertices1, vertex_hbond, masif_opts["feature_interpolation"])
        vertex_hphobicity = assignChargesToNewMesh(regular_mesh.vertices, vertices1, vertex_hphobicity, masif_opts["feature_interpolation"])
        
        # Step6: Compute APBS
        vertex_charges = computeAPBS(regular_mesh.vertices, tmp_protonated_base+".pdb", tmp_protonated_base)

        # Step7: construct the mesh
        mesh = create_mesh(regular_mesh.vertices, regular_mesh.faces, 
                        normals=vertex_normal, charges=vertex_charges, 
                        hbond=vertex_hbond, hphob=vertex_hphobicity,
                        normalize_charges=True)
        
        # Step8: compute the features
        vertice_features = compute_features(mesh)

        # Step9: clean up temp files
        shutil.rmtree(TMP_FOLDER)

        return (uniprot_id, vertice_features)
    except Exception as e:
        print(f"Error processing entry {uniprot_id}: {e}")
        return (uniprot_id, None)
        

def save_batch_to_arrow(batch_results, batch_path):
    combined_data = {
        "uniprot_id": [result[0] for result in batch_results],
        "vertices_features": [result[1].tolist() if result[1] is not None else None for result in batch_results]
    }
    batch_dataset = Dataset.from_dict(combined_data)
    batch_dataset.save_to_disk(batch_path)

# Path to the dataset folder inside Apptainer, unified via bind
dataset_path = "/mnt/afdb_datasets"
# Find all Arrow files in the dataset folder
arrow_files = sorted(glob.glob(f"{dataset_path}/chunk_*/data-*.arrow"))  # Ensure correct order

N_DATA = len(arrow_files)
N_TOTAL_JOBS = int(sys.argv[1])
JOB_ID = int(sys.argv[2]) - 1
N_CPUS = int(os.getenv("SLURM_CPUS_PER_TASK", os.cpu_count()))
N_WORKERS = max(1, N_CPUS // 2)
start = (JOB_ID * N_DATA) // N_TOTAL_JOBS
end = min(((JOB_ID + 1) * N_DATA) // N_TOTAL_JOBS, N_DATA)
subset_arrows = arrow_files[start:end]  # This is a list of paths
subset_uniprot_ids, subset_strucutres = read_multiple_arrow_files(subset_arrows)
assert len(subset_uniprot_ids) == len(subset_strucutres)
print(f"JOB ID: {JOB_ID}, N_WORKERS: {N_WORKERS}, N_TOTAL_JOBS: {N_TOTAL_JOBS}, start: {start}, end: {end}, total_dataset_this_job: {len(subset_strucutres)}", flush=True)

batch_size = 1000
file_path = f'/mnt/masif_outputs/masif_vertices_features_job_{JOB_ID}'

batch_index = 0
TOTAL_BATCHES = len(subset_uniprot_ids) // batch_size + 1
for i in range(0, len(subset_uniprot_ids), batch_size):
    batch_start = i
    batch_end = min(i+batch_size, len(subset_uniprot_ids))
    batch_path = file_path + f"/job_{JOB_ID}_{batch_index}"
    if  os.path.exists(batch_path):
        print(f"JOBID {JOB_ID} batch {batch_index}/{TOTAL_BATCHES}, {batch_start}:{batch_end}, already exists, skipping...", flush=True)
        batch_index += 1
        continue
    batch_results = []
    batch_uniprotids = subset_uniprot_ids[batch_start:batch_end]
    batch_pdbs = subset_strucutres[batch_start:batch_end]
    with ProcessPoolExecutor(max_workers=N_WORKERS) as executor:
        futures = [executor.submit(process_entry, entry_name, entry_pdb_array) for entry_name, entry_pdb_array in zip(batch_uniprotids, batch_pdbs)]
        for future in tqdm(futures):
            batch_results.append(future.result())
    save_batch_to_arrow(batch_results, batch_path)
    print(f"JOBID {JOB_ID} batch {batch_index}/{TOTAL_BATCHES}, {batch_start}:{batch_end}, done...", flush=True)
    batch_index += 1