import ROOT
import awkward as ak
import numpy as np
import vector
import glob
import tqdm
import os
from array import array


CLASS_LABELS_COCOA = [0, 211, 130, 22, 11, 13]
CLASS_NAMES_COCOA = [
    r"none",
    r"ch.had",
    r"n.had",
    r"$\gamma$",
    r"$e^\pm$",
    r"$\mu^\pm$",
]

def get_class_names(dataset_name):
    if dataset_name.startswith("cocoa_"):
        return CLASS_NAMES_COCOA
    else:
        raise Exception("Unknown dataset name: {}".format(dataset_name))


def load_eval_data(path, max_files=None):
    yvals = []
    filenames = []
    filelist = list(glob.glob(path))
    if max_files is not None:
        filelist = filelist[:max_files]

    for fi in tqdm.tqdm(filelist):
        dd = ak.from_parquet(fi)
        yvals.append(dd)
        filenames.append(fi)

    data = ak.concatenate(yvals, axis=0)
    X = data["inputs"]

    yvals = {}
    for typ in ["gen", "pred"]:
        for k in data["particles"][typ].fields:
            yvals["{}_{}".format(typ, k)] = data["particles"][typ][k]

    for typ in ["gen", "pred"]:

        # Compute phi, px, py, pz
        yvals[typ + "_phi"] = np.arctan2(yvals[typ + "_sin_phi"], yvals[typ + "_cos_phi"])
        yvals[typ + "_px"] = yvals[typ + "_pt"] * yvals[typ + "_cos_phi"]
        yvals[typ + "_py"] = yvals[typ + "_pt"] * yvals[typ + "_sin_phi"]
        yvals[typ + "_pz"] = yvals[typ + "_pt"] * np.sinh(yvals[typ + "_eta"])

        # Get the jet vectors
        jetvec = vector.awk(data["jets"][typ])
        for k in ["pt", "eta", "phi", "energy"]:
            yvals["jets_{}_{}".format(typ, k)] = getattr(jetvec, k)

    for typ in ["gen", "pred"]:
        for val in ["pt", "eta", "sin_phi", "cos_phi", "energy"]:
            yvals["{}_{}".format(typ, val)] = yvals["{}_{}".format(typ, val)] * (yvals["{}_cls_id".format(typ)] != 0)
            
    event_ids = data["event_id"]
    file_ids = data["file_id"]


    return yvals, X, filenames, event_ids, file_ids

def compute_met_and_ratio(yvals):
    msk_gen = yvals["gen_cls_id"] != 0
    gen_px = yvals["gen_px"][msk_gen]
    gen_py = yvals["gen_py"][msk_gen]

    msk_pred = yvals["pred_cls_id"] != 0
    pred_px = yvals["pred_px"][msk_pred]
    pred_py = yvals["pred_py"][msk_pred]

#     msk_cand = yvals["cand_cls_id"] != 0
#     cand_px = yvals["cand_px"][msk_cand]
#     cand_py = yvals["cand_py"][msk_cand]

    gen_met = awkward.to_numpy(np.sqrt(np.sum(gen_px, axis=1) ** 2 + np.sum(gen_py, axis=1) ** 2))
    pred_met = awkward.to_numpy(np.sqrt(np.sum(pred_px, axis=1) ** 2 + np.sum(pred_py, axis=1) ** 2))
#   cand_met = awkward.to_numpy(np.sqrt(np.sum(cand_px, axis=1) ** 2 + np.sum(cand_py, axis=1) ** 2))

    met_ratio_pred = awkward.to_numpy(pred_met / gen_met)
#    met_ratio_cand = awkward.to_numpy(cand_met / gen_met)

    return {
        "gen_met": gen_met,
        "pred_met": pred_met,
        #"cand_met": cand_met,
        "ratio_pred": met_ratio_pred,
        #"ratio_cand": met_ratio_cand,
    }

def create_root_file(output_path, yvals,X, event_ids, file_ids):
    # Create a ROOT file
    root_file = ROOT.TFile(output_path, 'RECREATE')
    jets = ROOT.TTree('jets', 'Tree with jet data per event')
    parts = ROOT.TTree('parts', 'Tree with jet data per event')
    events = ROOT.TTree('events', 'Tree with jet data per event')
    files = ROOT.TTree('files', 'Tree with jet data per event')
   

    # Define branches
    gen_jet_pt = ROOT.std.vector('float')()
    gen_jet_eta = ROOT.std.vector('float')()
    gen_jet_phi = ROOT.std.vector('float')()
    pred_jet_pt = ROOT.std.vector('float')()
    pred_jet_eta = ROOT.std.vector('float')()
    pred_jet_phi = ROOT.std.vector('float')()

    jets.Branch('gen_jet_pt', gen_jet_pt)
    jets.Branch('gen_jet_phi', gen_jet_phi)
    jets.Branch('gen_jet_eta', gen_jet_eta)
    jets.Branch('pred_jet_pt', pred_jet_pt)
    jets.Branch('pred_jet_phi', pred_jet_phi)
    jets.Branch('pred_jet_eta', pred_jet_eta)


     # Define branches
    gen_pt = ROOT.std.vector('float')()
    gen_eta = ROOT.std.vector('float')()
    gen_phi = ROOT.std.vector('float')()
    gen_cl = ROOT.std.vector('float')()
    pred_pt = ROOT.std.vector('float')()
    pred_eta = ROOT.std.vector('float')()
    pred_phi = ROOT.std.vector('float')()
    pred_cl = ROOT.std.vector('float')()

    parts.Branch('gen_pt', gen_pt)
    parts.Branch('gen_phi', gen_phi)
    parts.Branch('gen_eta', gen_eta)
    parts.Branch('gen_cl', gen_cl)
    parts.Branch('pred_pt', pred_pt)
    parts.Branch('pred_phi', pred_phi)
    parts.Branch('pred_eta', pred_eta)
    parts.Branch('pred_cl', pred_cl)
    
    #Define branches for events and files
    file_id_value = array('i', [0])  # Integer array of length 1
    event_id_value = array('i', [0])
    
    files.Branch('file_id', file_id_value, 'file_id/I')
    events.Branch('event_id', event_id_value, 'event_id/I')

    max_events = max(len(yvals['jets_gen_pt']), len(yvals['jets_pred_pt']))

    # Loop over event
    for i in range(max_events):
    	gen_jet_pt.clear()
    	gen_jet_eta.clear()
    	gen_jet_phi.clear()

    	pred_jet_pt.clear()
    	pred_jet_eta.clear()
    	pred_jet_phi.clear()


    	# Fill generator-level jet properties
    	if i < len(yvals['jets_gen_pt']):
    		for pt, eta, phi in zip(yvals['jets_gen_pt'][i], yvals['jets_gen_eta'][i], yvals['jets_gen_phi'][i]):
    			gen_jet_pt.push_back(pt)
    			gen_jet_eta.push_back(eta)
    			gen_jet_phi.push_back(phi)

    	# Fill prediction-level jet properties
    	if i < len(yvals['jets_pred_pt']):
    		for pt, eta, phi in zip(yvals['jets_pred_pt'][i], yvals['jets_pred_eta'][i], yvals['jets_pred_phi'][i]):
    			pred_jet_pt.push_back(pt)
    			pred_jet_eta.push_back(eta)
    			pred_jet_phi.push_back(phi)


    	jets.Fill()


    max_events_pred = max(len(yvals['gen_pt']), len(yvals['pred_pt']))



    # Loop over event
    for i in range(max_events_pred):
    	gen_pt.clear()
    	gen_eta.clear()
    	gen_phi.clear()
    	gen_cl.clear()


    	pred_pt.clear()
    	pred_eta.clear()
    	pred_phi.clear()
    	pred_cl.clear()



    	# Fill generator-level jet properties
    	if i < len(yvals['gen_pt']):
    		for pt, eta, phi, cl in zip(yvals['gen_pt'][i], yvals['gen_eta'][i], yvals['gen_phi'][i], yvals['gen_cls_id'][i]):
    			gen_pt.push_back(pt)
    			gen_phi.push_back(phi)
    			gen_eta.push_back(eta)
    			gen_cl.push_back(cl)


    	# Fill prediction-level jet properties
    	if i < len(yvals['pred_pt']):
    		for pt, eta, phi, cl in zip(yvals['pred_pt'][i], yvals['pred_eta'][i], yvals['pred_phi'][i], yvals['pred_cls_id'][i]):
    			pred_pt.push_back(pt)
    			pred_phi.push_back(phi)
    			pred_eta.push_back(eta)
    			pred_cl.push_back(cl)



    	parts.Fill()
  
    	
    for file_id in file_ids:
    	file_id_value[0] = file_id
    	files.Fill()
    	
    for event_id in event_ids:
    	event_id_value[0] = event_id
    	events.Fill()
     
     

    # Write and close the file
    jets.Write()
    parts.Write()
    ids.Write()
    root_file.Close()


dataset_dir = "../particleflow/experiments/cocoa_20240109_161202_879750.wipp-gpu-wn243/evaluation/epoch_50/cocoa_edmjj_clusters_pf/"
path = os.path.join(dataset_dir, "*.parquet")
yvals, X, event_ids, file_ids, _ = load_eval_data(path)
# Create ROOT file
create_root_file('test_idxs_newtrain.root', yvals,X, event_ids, file_ids)
