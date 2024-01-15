#!/bin/bash
#PBS -q gpu
#PBS -N tfds_files_make
#PBS -l walltime=72:00:00
#PBS -l mem=10gb
#PBS -l ncpus=4
#PBS -l ngpus=1
#PBS -o output-tfds_cl_idx.log
#PBS -e error-tfds_cl_idx.log

# Source Conda script
source /usr/local/anaconda/3.8/etc/profile.d/conda.sh

# Create and activate a new Conda environment
conda create -n appenv -y
conda activate appenv

# Configure Conda channels
conda config --add channels conda-forge
conda config --set channel_priority strict

# Install Apptainer
conda install apptainer -y


IMG="https://hep.kbfi.ee/~joosep/tf-2.14.0.simg"
DATA_DIR="/storage/agrp/annai/NEW_HGFlow/MLPF/cocoa_ttbar_clusters_idxs_tensorflow/"
MANUAL_DIR="/storage/agrp/annai/NEW_HGFlow/MLPF/cocoa_ttbar_parquet_clusters_idxs/cocoa_ttbar_clusters_pf"

cd /storage/agrp/annai/NEW_HGFlow/MLPF/particleflow/

apptainer exec --nv -B /storage/agrp/annai/NEW_HGFlow/ $IMG\
		  tfds build mlpf/heptfds/cocoa_ttbar/jj.py \
		  --data_dir $DATA_DIR \
		  --manual_dir $MANUAL_DIR

