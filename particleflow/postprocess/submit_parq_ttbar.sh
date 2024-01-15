#!/bin/bash
#PBS -q gpu
#PBS -N test
#PBS -l walltime=72:00:00
#PBS -l mem=8gb
#PBS -l ncpus=4
#PBS -l ngpus=1
#PBS -o output-450.log
#PBS -e error-400.log


cd /srv01/agrp/annai/annai/NEW_HGFlow/MLPF/

input_base_path="/storage/agrp/dreyet/PFlow/samples/ttbar/Ntuples_ttbar_noID_withSmearing_newpflow_17122023/"
output_base_path="/srv01/agrp/annai/annai/NEW_HGFlow/MLPF/cocoa_ttbar_parquet_clusters_idxs/"



# Loop over the file numbers
for i in {401..450}  # Adjust 10 to the actual number of files you have
do
    input_file="${input_base_path}/cocoa_ttbar_${i}.root"
    output_file="${output_base_path}/cocoa_ttbar_${i}.parquet"

    # Run the Python script with the current pair of input and output files
    python3 process_cocoa_cl_idxs.py "$i" "$input_file" "$output_file"
done
