# MLPF_Cocoa

MLPF code was adopted for the COCOA datasets and with some other minor changes in order to use different validation datasets from the kids. 

The main MLPF code (ref inside the code) was changed to run on the COCOA dataset, including changes in the main `train.py` script, utils, processing, and so on. 

Additionally, new configuration files (parameters/cocoa.config) were created to match the new datasets.

The script for processing COCOA root files to parquet and TensorFlow datasets has also been added as a part of the pre-processing step. This can be found in the new folder `postprocessing/`.

The evaluated parquet files can also be converted back to ROOT format. The code for this process is also available.

## Step 1 - Convert ROOT to Parquet files

- **COCOA samples**: Use the script `postprocess/process_cocoa_cl_idxs.py`.
  - To run it on a cluster, use `submit_parq_dijets.sh`.

- **CLICK samples**: Use the script `scripts/fcc/postprocessing.py`.
  - To run it on a cluster, use `scripts/fcc/submit_click_modified.sh`.

## Step 2 - Convert Parquet to TensorFlow Datasets (TFDS)

- **COCOA samples**: Run `make_tfds.sh` with the correct Python file `mlpf/heptfds/cocoa_dijets/jj.py` (for dijet samples).
  - **Note**: If you want to make split samples, you must specify how you want to split the data in `mlpf/heptfds/cocoa_dijets/jj.py` using the function from `utils_edm.py` called `split_sample_train(test/valid)`.  
  - This is necessary because the same events must be used in HGPFlow for training/validation/testing.
  - If you don't care about this, you can simply use `split_sample`, which will automatically split the data into test and train (you can also add validation there).

- **CLICK samples**: Run `make_tfds.sh` with the correct Python file `click_pf_edm4hep/qq.py`. The same as for the COCOA samples you need to use correct split_sample otherwise you will be not able to run the training.

At the end of the day, you shall have for example, the following tfds folder: cocoa_samples/cocoa_edmjj_clusters_pf cocoa_hbb_boost_clusters_pf  cocoa_hbb_resolve_clusters_pf  cocoa_pions_clusters_pf  cocoattbar_clusters_pf (those are all cocoa files which you will be using not only for training but also for testing). Inside each folder, there shall be 1.7.0/cocoa_edmjj_clusters_pf-train.array_record-000XX-of-000YY cocoa_edmjj_clusters_pf-test.array_record-000XX-of-000YY cocoa_edmjj_clusters_pf-validation.array_record-000XX-of-000YY dataset_info.json features.json metadata.json 

However, given that you create the test/validation/training sample separately using make_tfds.sh, your task is to combine them in one folder: cocoa_edmjj_clusters_pf/1.7.0/, where the dataset_info.json will have ALL the samples and their sizes. Please see the example in my folder: /storage/agrp/annai/NEW_HGFlow/MLPF_newClick_train_modified/datacocoa_new_test cocoa_new_train cocoa_new_valid are then combined in folder cocoa_new. 

## Step 3 - Run the Training

- This can be done using the shell scripts `submit_train_click.sh` or `submit_train_cocoa.sh`.
- Update the paths and the data files as needed. 
- **Note**: If you have a different number of validation events, you need to specify them in the `parameters/cocoa.yaml` or `click.yaml` file under the line `"validation_num_events:"`.

## Step 4 - Run the Predictions

- Use the `submit_eval_click.sh` or `submit_eval_cocoa.sh` scripts.
- If you want to evaluate other samples, specify them in the parameter file, as shown here:  
  [parameters/cocoa.yaml#L234](https://github.com/annaivina/MLPF_cocoa/blob/new_dev/particleflow/parameters/cocoa.yaml#L234).
