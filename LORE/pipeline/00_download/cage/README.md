# FANTOM5 CAGE peaks

Source for the `cage` modality: Cap Analysis of Gene Expression peaks, which mark
transcription start sites genome-wide. TPM-normalized, from the FANTOM5 project at RIKEN.

Base URL: <https://fantom.gsc.riken.jp/5/datafiles/phase2.6/extra/CAGE_peaks>

> ⚠ FANTOM5's data licence is **not stated** in the RIKEN `datafiles/` tree — the
> accompanying publications are CC-BY-4.0, which does not by itself license the data.
> Confirm terms with RIKEN before redistributing anything derived from it. See
> `LORE/NOTICE.md`.

## Files

Six species/assemblies were used. Note that human and mouse use the combined
phase-1-and-2 files, while the others are plain phase-2:

| species | assembly | file |
|:-|:-|:-|
| Human | hg19 | `hg19.cage_peak_phase1and2combined_tpm.osc.txt.gz` |
| Mouse | mm9 | `mm9.cage_peak_phase1and2combined_tpm.osc.txt.gz` |
| Rat | rn6 | `rn6.cage_peak_tpm.osc.txt.gz` |
| Rhesus macaque | rheMac8 | `rheMac8.cage_peak_tpm.osc.txt.gz` |
| Chicken | galGal5 | `galGal5.cage_peak_tpm.osc.txt.gz` |
| Dog | canFam3 | `canFam3.cage_peak_tpm.osc.txt.gz` |

## Download

```bash
DL_DIR=$LORE_DATA_ROOT/data/downloads/cage
BASE_URL=https://fantom.gsc.riken.jp/5/datafiles/phase2.6/extra/CAGE_peaks
mkdir -p "$DL_DIR" && cd "$DL_DIR"

for F in hg19.cage_peak_phase1and2combined_tpm.osc.txt.gz \
         mm9.cage_peak_phase1and2combined_tpm.osc.txt.gz \
         rn6.cage_peak_tpm.osc.txt.gz \
         rheMac8.cage_peak_tpm.osc.txt.gz \
         galGal5.cage_peak_tpm.osc.txt.gz \
         canFam3.cage_peak_tpm.osc.txt.gz ; do
    wget -c "$BASE_URL/$F"
    gunzip -f "$F"
done
```

Note that hg19 and mm9 are **older assemblies** than the rest of the pipeline uses;
`../../01_process/cage/` lifts the peaks over before extracting per-transcript scores.

## Next

`../../01_process/cage/` converts peaks to BigWig, pairs them with transcript
coordinates, and extracts per-transcript CAGE tracks. Each track is paired with a
free-text descriptor of the sample it came from, carried as the `context` modality, so
`cage` is condition-conditional.
