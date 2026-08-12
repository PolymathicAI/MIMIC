
from lore import paths
import os

intermediate_path = paths.get_path(data_type="data", stage="intermediate", name="refseq", version="229")

gcf_dirs = [d for d in os.listdir(intermediate_path) if d.startswith("GCF") and os.path.isdir(intermediate_path / d)]

# gcf_dirs = gcf_dirs[0:2]

function = "02_compile_isoform_annotations"
    
outfile = os.path.join(os.getcwd(), f"taskfile_{function}.txt")

with open(outfile, "w") as f:
    for genome in gcf_dirs:
        f.write(f"(source ../../../venvs/central/bin/activate ; python {function}.py --genome {genome} --overwrite) &> {os.path.join(intermediate_path, genome)}/{function}.log\n")
