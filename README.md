# MLPF_cocoa
MLPF code adopted for the COCOA datasets 

The main MLPF code (ref inside the code) was changed to run on the cocoa dataset, including changes in the main train.py script, utils, processing, and so on. 

Additionally, new configuration files are created to match the new datasets 

The script for processing COCOA root files to parquet to TensorFlow datasets has also been added as a part of the pre-processing step. Located in a new folder postprocessing/

The evaluated files (parquet) file can also be changed to root style format. The code for such things is also availble 

# Step 1 - Make root --> parquet file 
COCOA samples - postprocess/process_cocoa_cl_idxs.py is the main script\
To run it on cluster use submit_parq_dijets.sh\

CLICK samples - scripts/fcc/postprocessing.py is the main script\
To run it on cluster use scripts/fcc/submit_click_modified.sh

#Step 2 - Make parquet --> tfds files 
COCOA samples - make_tfds.sh 
NB: if you want to make slit samples you must specify how you want to split in mlpf/heptfds/cocoa_dijets/jj.py (
