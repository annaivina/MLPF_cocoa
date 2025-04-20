#!/bin/bash
#PBS -q gpu
#PBS -N tfds_dijets
#PBS -l walltime=10:00:00
#PBS -l mem=10gb
#PBS -l ncpus=1
#PBS -l ngpus=1
#PBS -l io=2
#PBS -o output-tfds_dijets_test.log
#PBS -e error-tfds_dijets_test.log

source /usr/wipp/conda/24.5.0u/etc/profile.d/conda.sh
conda activate apptainer
cd /srv01/agrp/annai/annai/NEW_HGFlow/MLPF_newClick_train_modified/particleflow/


IMG="/storage/agrp/annai/6eeeaf4590c9bdcc3af03e2e5b37f78b1bf59b164317ca3ab7f89adcd555eb53"
DATA_DIR="/storage/agrp/annai/NEW_HGFlow/MLPF_newClick_train_modified/data/tdfs_dijets_test"
MANUAL_DIR="/srv01/agrp/annai/annai/NEW_HGFlow/MLPF/cocoa_dijet_parquet_clusters_idxs_debug_new_samples_train_val_ALL/test/"



apptainer exec --nv -B /storage/agrp/annai/NEW_HGFlow/ $IMG\
		  tfds build mlpf/heptfds/cocoa_dijets/jj.py \
		  --data_dir $DATA_DIR \
		  --manual_dir $MANUAL_DIR

