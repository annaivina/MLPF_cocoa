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

# Step 2 - Make parquet --> tfds files 
COCOA samples - run make_tfds.sh with correct python file mlpf/heptfds/cocoa_dijets/jj.py (for dijet sample) 
NB: if you want to make split samples you must specify how you want to split in mlpf/heptfds/cocoa_dijets/jj.py using function from utils_ed.py split_sampletrain(test/valid)\
The reason for this is because you need to have exactly the same events are trained in HGPFlow\
If you dont  are about it you just use split_sample - which it will automatically split to test and train (you can also add validation there)

CLICK samples - run make_tfds.sh  with correct python file click_pf_edm4hep/qq.py 

# Step 3 - Run the training 
This can be done using two main shell scripts submit_train_click.sh or submit_train_cocoa.sh\
You just need to change the paths and the data files. \
NB: If you have different number of the validationb events you need to specify then in parameters/cocoa.yaml or click.yaml file in the line "validation_num_events: "

# Step 4 - run the predictions 
Done using submit_eval_click.sh or submit_eval_cocoa.sh files\
If you want to have other samples to evaluate on you need to specify them also in the parameter card like here: https://github.com/annaivina/MLPF_cocoa/blob/new_dev/particleflow/parameters/cocoa.yaml#L234\
