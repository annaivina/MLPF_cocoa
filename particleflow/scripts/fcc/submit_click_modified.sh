#!/bin/bash
#PBS -N make_parq-click
#PBS -l walltime=72:00:00
#PBS -l mem=30gb
#PBS -l io=4
#PBS -l ncpus=1
#PBS -o output-parq_click14.log
#PBS -e error-parq_click14.log


source /usr/wipp/conda/24.5.0u/etc/profile.d/conda.sh
conda activate apptainer
cd /srv01/agrp/annai/annai/NEW_HGFlow/MLPF_newClick_train_modified/particleflow/

# #Submit train_val
#input_base_path="/storage/agrp/dreyet/FullEventReco/HGPflow/clic/train_samples/train_val_mlpf_150k/"
#input_base_path="/storage/agrp/dreyet/FullEventReco/HGPflow/clic/train_samples/train_val_mlpf_350k"
#output_base_path="/storage/agrp/annai/NEW_HGFlow/MLPF_newClick_train_modified/data/click_train_val/extended/"

#Submit test
input_base_path="/storage/agrp/dreyet/FullEventReco/HGPflow/clic/train_samples/test_mlpf_20k/"
output_base_path="/storage/agrp/annai/NEW_HGFlow/MLPF_newClick_train_modified/data/click_test"

IMG=/storage/agrp/annai/6eeeaf4590c9bdcc3af03e2e5b37f78b1bf59b164317ca3ab7f89adcd555eb53


for i in 14
do 
	input_file="${input_base_path}/merged_chunk_${i}.root"
	output_file="${output_base_path}/train_${i}.parquet"
	apptainer exec -B /storage/agrp/ $IMG  python3.10 scripts/fcc/postprocessing.py "$i" "$input_file" "$output_file"
done
