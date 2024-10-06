#!/bin/bash
#PBS -N mlpf_cocoa_data
#PBS -l walltime=72:00:00
#PBS -l mem=8gb
#PBS -l ncpus=4
#PBS -l io=5
#PBS -o output-pions-4-5.log
#PBS -e error-pions-4-5.log

source /usr/wipp/conda/24.5.0u/etc/profile.d/conda.sh
conda activate apptainer
cd /srv01/agrp/annai/annai/NEW_HGFlow/MLPF/

# input_base_path="/storage/agrp/dreyet/PFlow/samples/dijet/Ntuples_dijet_noID_withSmearing_newpflow_15122023/"
# output_base_path="/srv01/agrp/annai/annai/NEW_HGFlow/MLPF/cocoa_dijet_parquet_clusters_idxs_debug_new_samples_train_val_ALL/"

# input_base_path="/storage/agrp/dreyet/PFlow/samples/higgs/Ntuples_ZHbb_pTHatMin100_noID_withSmearing_newpflow_20112023"
# output_base_path="/srv01/agrp/annai/annai/NEW_HGFlow/MLPF/cocoa_zhbb_boosted_parquet_clusters_idxs_debug_new_samples_test/"

input_base_path="/storage/agrp/dreyet/PFlow/samples/singleParticle/Ntuples_charged_pion_noID_withSmearing_newpflow_29092024/pt_binning/"
output_base_path="/srv01/agrp/annai/annai/NEW_HGFlow/MLPF/cocoa_pions_boosted_parquet_clusters_idxs_debug_new_samples_test/"



IMG=/storage/agrp/annai/6eeeaf4590c9bdcc3af03e2e5b37f78b1bf59b164317ca3ab7f89adcd555eb53

# Loop over the file numbers
for i in {3..4}
do
    #input_file="${input_base_path}/cocoa_dijet_${i}_skim.root"
    input_file="${input_base_path}/merged_bin_${i}_allEta_skim.root"
    #input_file="${input_base_path}test_skim.root"
    output_file="${output_base_path}/merged_bin_${i}_allEta_skim.parquet"
    #output_file="${output_base_path}/test_skim.parquet"

    # Run the Python script with the current pair of input and output files
    apptainer exec -B /storage/agrp/ $IMG  python3.10 process_cocoa_cl_idxs.py "$i" "$input_file" "$output_file"
done
