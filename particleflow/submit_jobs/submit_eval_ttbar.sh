#!/bin/bash
#PBS -q gpu
#PBS -N tests_mlpf
#PBS -l walltime=72:00:00
#PBS -l mem=8gb
#PBS -l ncpus=4
#PBS -l ngpus=1
#PBS -o output-cocoa_eval.log
#PBS -e error-cocoa_eval.log

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


IMG=https://hep.kbfi.ee/~joosep/tf-2.14.0.simg

cd /storage/agrp/annai/NEW_HGFlow/MLPF/particleflow/

apptainer exec -B /storage/agrp/annai/NEW_HGFlow/ --nv \
    --env PYTHONPATH=hep_tfds \
    --env TFDS_DATA_DIR=/storage/agrp/annai/NEW_HGFlow//MLPF/cocoa_dijet_clusters_idxs_tensorflow/ \
    $IMG python3.10 mlpf/pipeline.py evaluate --train-dir experiments/cocoa_20240109_161202_879750.wipp-gpu-wn243/
