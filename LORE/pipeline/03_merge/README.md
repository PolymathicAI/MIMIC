This folder performs the merge according to the config yaml file. It assumes that the id modality includes dataset,train,test,val.parquet files, where dataset is the combination of the different splits.

In order to use this, run 00, 01, then 02, etc... When running 01, a decision about which subsets to include needs to be taken in the form of a number k which specifies the top subsets to pick.

The 02 merge script was run via:
sbatch --job-name=02_merge --cpus-per-task=64 --mem=1500G -t 1-00:00:00 --wrap="python -u 02_merge_script.py"
The 04 tokenize script is run via:
sbatch --job-name=04_tokenize --cpus-per-task=96 --mem=1500G -t 1-00:00:00 --wrap="python -u 04_tokenize_dataset.py"
The 05 webdataset conversion is run via:
sbatch --job-name=05_webdataset --cpus-per-task=96 --mem=1500G -t 1-00:00:00 --wrap="python -u 05_write_webdataset.py"

The others can be run locally.

Note that 04_tokenize_dataset.py will create a new config yaml with the tokenizer versions. This is used in the downstream scripts.