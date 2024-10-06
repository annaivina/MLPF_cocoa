#!/bin/bash
#PBS -q gpu
#PBS -N mlpf_train_150k_300epoch
#PBS -l walltime=72:00:00
#PBS -l mem=20gb
#PBS -l ncpus=2
#PBS -l ngpus=1
#PBS -l io=4
#PBS -o output-clic150_300epoch.log
#PBS -e error-clic150_300epoch.log

# Source Conda script
source /usr/wipp/conda/24.5.0u/etc/profile.d/conda.sh
conda activate apptainer
cd /srv01/agrp/annai/annai/NEW_HGFlow/MLPF_newClick_train_modified/particleflow/

export CUDA_VISIBLE_DEVICES=0

IMG=/storage/agrp/annai/6eeeaf4590c9bdcc3af03e2e5b37f78b1bf59b164317ca3ab7f89adcd555eb53


apptainer exec -B /storage/agrp/annai/NEW_HGFlow/ --nv \
    --env PYTHONPATH=hep_tfds \
    --env TFDS_DATA_DIR=/storage/agrp/annai/NEW_HGFlow/MLPF_newClick_train_modified/data/data_tfds_150k \
    $IMG python3.10 mlpf/pipeline.py train -c parameters/clic.yaml \
    --plot-freq 1 --batch-multiplier 0.5
