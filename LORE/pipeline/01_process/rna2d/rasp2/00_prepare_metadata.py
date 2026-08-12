"""
This script prepares the metadata for the RASP2 RNA structure processing.
It parses the filename of the RASP2 score data and extracts the relevant information.
The metadata is saved in a CSV file.

Usage:
python prepare_metadata.py

Output:
The metadata is saved in the `score_data` directory.
"""
#%%
import pandas as pd
import re, glob, os
from typing import List, Dict, Optional
from species_genome_dict import SPECIES_TO_GENOME_DICT
from lore.paths import get_path

# Combined list of all valid reagents from all species/datasets
VALID_REAGENTS = [
    'NAI', 'B5', 'NAI-N3', 'RNase', '6A3', 'DMS', '5NIA', '2A3', 'V1_S1',
    'NIC', 'N3-kethoxal', '1M7', 'V1&S1', 'I5', 'Naz-N3', '1M4',
    '1M6', 'DMS&CMCT', 'NMIA', 'DMS_CMCT', 'SHAPE-MaP', 'V1', 'BzCN',
    'hydroxyl_radical', 'lead(II)'
]

# Combined list of all valid conditions from all species/datasets
# Note: All spaces will be converted to underscores to match filename format.
# This list will be sorted by length in descending order before use in the parser.
VALID_CONDITIONS = [
    "ex vivo", "invivo_score", "in vivo", "native & deproteined", "in vivo 1ng",
    "kethoxal_vs_no-treat", "vivo", "denatured", "kethoxal_vs_no-treat_invitro-PDS",
    "invitro", "vitro", "native___deproteined___renatured", "native___deproteined",
    "cytoplasm_invitro", "kethoxal_vs_no-treat_invitro", "nucleoplasm_invivo",
    "chromatin_invivo", "native & deproteined & renatured", "Dicer-enriched RNA",
    "_ex_vivo", "invivo_K", "kethoxal_vs_no-treat_PDS", "renatured",
    "chromatin_invitro", "nucleoplasm_invitro", "invivo", "in vivo 25ng",
    "cytoplasm_invivo", "in vivo 125ng", "in vivo 5ng", "invivo_Na",
    "in vitro transcribed cotranscriptional", "in vitro transcribed equilibrium ",
    "in vitro transcribed", "in vitro", "invivo_DMS_CMCT", "kethoxal_vs_no-treat-5min",
    "exvivo", "kethoxal_vs_no-treat-invivo", "kethoxal_vs_no-treat-invitro",
    "invivo_Li", "invitro_K150", "kethoxal_vs_no-treat-1min", "invivo_rep1",
    "invitro_K0", "invivo_METTL3_KO", "invitro_95degree", "incell", "invivo_rep2",
    "invivo_vs_kethoxal-remove", "kethoxal_vs_kethoxal-remove", "invivo_PDS",
    "invitro_KO", "shield_in_vivo", "shield_elavl1a-mo_in_vivo", "WT_shield_invivo",
    "4-cell_in_vivo", "64-cell_in_vivo", "WT_64c_invitro",
    "65c_translation_inhibitors_PatA_treated", "sphere_elavl1a-mo_in_vivo",
    "WT_64c_invivo", "fertilized_egg_in_vivo", "sphere_in_vivo", "1-cell_in_vivo",
    "64c_translation_inhibitors_CHX_treated", "64c_translation_inhibitors_PatA_treated",
    "WT_64c_translation_inhibitors_untreated", "WT_sphere_invivo",
    "purified_18S_ribosomal_RNA", "WT", "in_vivo", "L26_Delta_invivo", "37_degree",
    "TGIRT_DMSvivo_rep1", "75_degree", "SSii_untreated", "WT_invivo", "23_degree",
    "TGIRT_DMSvivo_additionalSeq", "purified_20S_ribosomal_RNA", "purified_mature_40S",
    "SSii_DMSvivo_rep2", "TGIRT_untreated", "TGIRT_DMSvivo_rep2", "55_degree",
    "Ded1p_Mutate", "purified_pre-40S", "30_degree", "SSii_DMSvivo_rep1",
    "2andHalf_percent_DMSvivo", "5percent_DMSvivo", "deproteinized",
    "in vivo NAI", "in vitro DMS", "in_vitro_NAI",
    "37degree_invitro_WT", "37degree_invivo_WT1", "37degree_invivo_WT2",
    "37degree_invivo_deltagcvB_Ksg", "37degree_invivo_deltagcvB",
    "37degree_invivo_dusB-M3_mutant2", "37degree_invivo_dusB-M3_mutant",
    "95degree_invitro_WT", "Cellfree", "Kasugamycin",
    "DMS_vitro_K0", "DMS_vitro_K150", "25degree_invivo", "37degree_invivo",
    "extracted from virions", "plant cell lysates", "exvirion", "refolded",
    "exviron", "inviron", "invivo_1M6", "invivo_1M7", "invivo_NMIA",
    "in virio", "in vitro transcribed RNA", "naked viral RNA", "invirio",
    "in vitro 20uM NSP2", "in vitro 0uM NSP2", "in vitro 10uM NSP2", "in vitro 5uM NSP2",
    "exvitro"
]

# Combined list of all valid cell lines from all species/datasets
# Note: Spaces and special characters are converted to underscores where needed.
VALID_CELL_LINES = [
    'GM12891', 'A549', 'HEK293', 'HELA', 'H9', 'HEPG2',
    'HEK293T', 'GM12878', 'GM12892', 'K562', 'FBL',
    'MEF', 'C57BL_6J_mouse_Ly6Clo_macrophages', 'C57BL_6J_mouse_Ly6Chi_macrophages',
    'TSC', 'mES', 'trophoblast_stem_cells',
    'widetype_AB_strain', 'embryonic_cells', 'Vero',
    'BY4741_S15_YRR_mutant', 'rpl26delta', 'S288C', 'BY4741',
    'BY4741_Rio2_dloop_mutant', 'yRH101', 'dbp2delta',
    'seeding', 'seedling', 'delta382', 'USA-WA1', 'Wuhan-Hu-1', 'Leiden-0002'
]


def parse_filename_with_genome(filename: str, genome: str, species: str, kingdom: str) -> Optional[Dict[str, str]]:
    # --- Prepare patterns from the definitive lists ---

    # 1. Reagent pattern: Escape each item to handle special chars like '&'
    #    and add 'none' as a special case.
    reagent_options = [re.escape(r) for r in VALID_REAGENTS]
    reagent_pattern = f"(?i:{'|'.join(reagent_options)}|none)"

    # 2. Condition pattern: Replace spaces with underscores, escape special chars.
    #    CRITICAL: Sort by length (desc) to match longer strings first.
    condition_options_transformed = [c.replace(' ', '_') for c in VALID_CONDITIONS]
    condition_options_transformed.sort(key=len, reverse=True)
    condition_options_escaped = [re.escape(c) for c in condition_options_transformed]
    condition_pattern = f"(?i:{'|'.join(condition_options_escaped)})"

    # 3. Cell line pattern (unchanged)
    cell_line_pattern = f"(?i:{'|'.join(VALID_CELL_LINES)}|none)"

    # --- Build the final regex ---
    pattern_string = (
        r"^(?P<technology>[\w-]+)_"
        f"{re.escape(genome)}_"
        r"(?P<publication>.+?)_"
        r"(?P<year>\d{4})_"
        f"(?P<reagent>{reagent_pattern})_"
        f"(?P<condition>{condition_pattern})_"
        r"(?P<strand>plus|minus|both)_"
        r"(?P<scale>transcriptome-wide|targeted)"
        f"(?:_(?P<cellline>{cell_line_pattern}))?"
        r"[_\\-](?P<description>.*)$"
    )

    pattern = re.compile(pattern_string)
    match = pattern.match(filename)
    if not match:
        return None

    data = match.groupdict()

    # --- Post-processing ---
    data['genome'] = genome
    data['publication'] = data['publication'].replace('_', ' ')
    data["sepecies"] = species
    data['kingdom'] = kingdom
    data["file_name"] = filename

    # Standardize reagent
    captured_reagent = data.get('reagent')
    if captured_reagent and captured_reagent.lower() == 'none':
        data['reagent'] = 'N/A'

    # Standardize cell line
    captured_cell_line = data.get('cellline')
    if captured_cell_line and captured_cell_line.lower() != 'none':
        data['cellline'] = captured_cell_line.upper()
    else:
        data['cellline'] = 'N/A'

    return data

#%%
if __name__ == "__main__":
    root_path = get_path("data", "downloads", "rna2d", "rasp2")
    for kingdom in SPECIES_TO_GENOME_DICT.keys():
        for species in SPECIES_TO_GENOME_DICT[kingdom].keys():
            paths = glob.glob(f"{str(root_path)}/score_data/{kingdom}/RASP_files_{species}/*.bw")
            paths.sort()
            dict_list = []
            for path in paths:
                # Extract the filename from the path
                filename = os.path.basename(path)
                # Parse the filename with the genomeb
                parsed_data = parse_filename_with_genome(filename, SPECIES_TO_GENOME_DICT[kingdom][species], species, kingdom)
                if parsed_data:
                    dict_list.append(parsed_data)
                else:
                    print(f"Failed to parse filename: {filename}")
            df = pd.DataFrame(dict_list)
            df.to_csv(f"{str(root_path)}/score_data/{kingdom}/metadata_{species}_rasp2.csv", index=False)
# %%
