#!/bin/bash
#PBS -q gpu
#PBS -N mlpf_test_300epoch
#PBS -l walltime=10:00:00
#PBS -l mem=8gb
#PBS -l ncpus=4
#PBS -l ngpus=1
#PBS -l io=4
#PBS -o output-clic_test300epoch.log
#PBS -e error-clic_test300epoch.log

# Source Conda script
source /usr/wipp/conda/24.5.0u/etc/profile.d/conda.sh
conda activate apptainer
cd /srv01/agrp/annai/annai/NEW_HGFlow/MLPF_newClick_train_modified/particleflow/

export CUDA_VISIBLE_DEVICES=0

IMG=/storage/agrp/annai/6eeeaf4590c9bdcc3af03e2e5b37f78b1bf59b164317ca3ab7f89adcd555eb53


apptainer exec -B /storage/agrp/annai/NEW_HGFlow/ --nv \
    --env PYTHONPATH=hep_tfds \
    --env TFDS_DATA_DIR=/storage/agrp/annai/NEW_HGFlow/MLPF_newClick_train_modified/data/data_tfds_150k \
    $IMG python3.10 mlpf/pipeline.py evaluate --train-dir experiments/clic_20240829_143930_946747.gwn246/
