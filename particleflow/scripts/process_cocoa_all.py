import numpy as np
import awkward
import uproot
import glob
import os
import sys
import multiprocessing
from scipy.sparse import coo_matrix


track_feature_order = [
    "elemtype",
    "pt",
    "eta",
    "sin_phi",
    "cos_phi",
    "p",
    "chi2",
    "ndf",
    "radiusOfInnermostHit",
    "tanLambda",
    "D0",
    "omega",
    "Z0",
    "time",
    "type",
]
cell_feature_order = [
    "elemtype",
    "et",
    "eta",
    "sin_phi",
    "cos_phi",
    "energy",
    "position.x",
    "position.y",
    "position.z",
    "time",
    "subdetector",
    "type",
]


# the feature matrices will be saved in this order
particle_feature_order = ["PDG", "charge", "pt", "eta", "sin_phi", "cos_phi", "energy"]

def sanitize(arr):
    arr[np.isnan(arr)] = 0.0
    arr[np.isinf(arr)] = 0.0



def gen_to_features(genpart_data, iev):
	feats_from_gp = ['particle_pdgid','particle_pt','particle_eta','particle_phi','particle_e']

	rename_map = {
		'particle_pdgid': 'PDG',
		'particle_e': 'energy'
		}

	ret = {rename_map.get(feat, feat.split('_')[1]): genpart_data[feat][iev] for feat in feats_from_gp}

	ret["sin_phi"] = np.sin(ret["phi"])
	ret["cos_phi"] = np.cos(ret["phi"])


	#Calculating the charge of the particle
	pdg_to_charge = {
		   -3334:   1,  3334:  -1, # omegalol :D
            -3322:  0,  3322:  0, # xi0
            -3312:  1,  3312:  -1, # xi+-
            -3222: -1, 3222:  1, # sigma+
            -3122: 0 , 3122:  0, # lambda
            -3112: 1 , 3112:  -1, # sigma-
            -2212:  -1, 2212:  1, # proton
            -2112:  0, 2112:  0, # neutrons
             -321: -1 ,  321:  1, # kaon+-
             -211:  -1,  211:  1, # pi+-
             -111: 0 ,  111:  0, # pi0
              -16: 0,   16: 0, # nu_tau
              -14: 0,   14: 0, # nu_mu
              -13: 1 ,   13: -1 , # mu
              -12: 0,   12: 0, # nu_e
              -11: 1,   11:  -1, # e
              -22:  0, 22:  0, # photon
              130:  0, -130:  0, # K0L
              310:  0, -310:  0, # K0S
	}

	# Calculate and add the charge for each particle in the event
	charges = [pdg_to_charge.get(pid, 0) for pid in ret['PDG']]
	ret['charge'] = charges

	unique_values = np.unique(ret['charge'])


	return awkward.Record(ret)


def track_to_features(track_data, iev):
	feats_from_track = ['track_phi','track_theta','track_qoverp','track_d0','track_z0']

	rename_map = {
		'track_z0': 'Z0',
        'track_d0': 'D0'}

	ret = {rename_map.get(feat, feat.split('_')[1]): track_data[feat][iev] for feat in feats_from_track}

	n_tr = len(ret["D0"])

	tt = np.tan(ret["theta"] / 2.0)
	eta = awkward.to_numpy(-np.log(tt, where=tt > 0))
	eta[tt <= 0] = 0.0
	ret["eta"] = eta


	ret["sin_phi"] = np.sin(ret["phi"])
	ret["cos_phi"] = np.cos(ret["phi"])

	p = 1/ret["qoverp"]
	ret["p"] = p
	ret["pt"]= p * np.sin(ret["theta"])


	new_elements = ['chi2', 'ndf','radiusOfInnermostHit','tanLambda','omega','time','type']
	for elem in new_elements:
		ret[elem] = np.zeros(n_tr, dtype=np.float32)


	# track is always type 1
	ret["elemtype"] = 1 * np.ones(n_tr, dtype=np.float32)

	return awkward.Record(ret)


def cell_to_features(cell_data, iev):
	feats_from_cell = ['cell_e','cell_eta','cell_phi','cell_x','cell_y','cell_z','cell_layer']

	rename_map = {
		'cell_x': 'position.x',
        'cell_y': 'position.y',
        'cell_z': 'position.z',
        'cell_e': 'energy'}

	ret = {rename_map.get(feat, feat.split('_')[1]): cell_data[feat][iev] for feat in feats_from_cell}

	n_cl = len(ret["eta"])

	theta = 2 * np.arctan(np.exp(-ret["eta"]))
	ret["et"]= ret["energy"] * np.sin(theta)

	ret["sin_phi"] = np.sin(ret["phi"])
	ret["cos_phi"] = np.cos(ret["phi"])

	#set the subdetector type
	# Apply conditions to each element in the 'layer' array
	# Layer 0-2 => subdetector type 0 (ecal)
	# Layer 3-5 => subdetector type 1 (hcal)
	# Layer > 5 => subdetector type 2 (other)

	sdcoll = "subdetector"
	ret[sdcoll] = np.where(ret["layer"] <= 2, 0, np.where(ret["layer"] <= 5, 1, 2))

	ret[sdcoll] = np.zeros(len(ret["eta"]), dtype=np.int32)
	ret[sdcoll] = np.where(ret["layer"] <= 2, 0, np.where(ret["layer"] <= 5, 1, 2))

	unique_values = np.unique(ret[sdcoll])



	new_elements = ['time','type']
	for elem in new_elements:
		ret[elem] = np.zeros(n_cl, dtype=np.float32)

	#cell is always type 2 (this is from their code )
	ret["elemtype"] = 2 * np.ones(len(ret["eta"]), dtype=np.int32)

	return awkward.Record(ret)




def particle_to_obj_map(cell_data,track_data,genp_data,iev):

	ngens = awkward.count(genp_data["particle_pdgid"][iev])

	#Initiate arrays
	particle_to_cell = -1 * np.ones(ngens, dtype=np.int32)
	particle_to_track = -1 * np.ones(ngens, dtype=np.int32)


	cell_tr_inx = cell_data['cell_parent_idx'][iev]
	particle_tr_idx = genp_data['particle_track_idx'][iev]


	for cell_idx, particle_idx in enumerate(cell_tr_inx):
		if particle_idx >= 0:
			particle_to_cell[particle_idx] = cell_idx


	particle_to_track[:] = particle_tr_idx


	particle_to_obj = -1 * np.ones((ngens, 2), dtype=np.int32)  # 2 columns: track, cell
	# Populate the array

	for i in range(ngens):
		track_idx = particle_to_track[i]
		cell_idx = particle_to_cell[i]


		if track_idx != -1:  # Prioritize track match
			particle_to_obj[i, 0] = track_idx
		elif cell_idx != -1:  # Use cell match if no track match
			particle_to_obj[i, 1] = cell_idx


	return particle_to_obj


def assign_to_recoobj(n_obj, obj_to_ptcl, used_particles):
    obj_to_ptcl_all = -1 * np.ones(n_obj, dtype=np.int64)
    for iobj in range(n_obj):
        if iobj in obj_to_ptcl:
            iptcl = obj_to_ptcl[iobj]
            obj_to_ptcl_all[iobj] = iptcl
            assert used_particles[iptcl] == 0
            used_particles[iptcl] = 1
    return obj_to_ptcl_all


def get_feature_matrix(feature_dict, features):
    feats = []
    for feat in features:
        feat_arr = awkward.to_numpy(feature_dict[feat])
        feats.append(feat_arr)
    feats = np.array(feats)
    return feats.T

def get_particle_feature_matrix(pfelem_to_particle, feature_dict, features):
    feats = []
    for feat in features:
        feat_arr = feature_dict[feat]
        if len(feat_arr) == 0:
            feat_arr_reordered = feat_arr
        else:
            feat_arr_reordered = awkward.to_numpy(feat_arr[pfelem_to_particle])
            feat_arr_reordered[pfelem_to_particle == -1] = 0.0
        feats.append(feat_arr_reordered)
    feats = np.array(feats)
    return feats.T

def map_neutral_to_charged(pdg):
    if pdg == 130 or pdg == 22:
        return 211
    return pdg

def map_charged_to_neutral(pdg):
    if pdg == 0:
        return 0
    if pdg == 11 or pdg == 22:
        return 22
    return 130

def map_pdgid_to_candid(pdgid, charge):
    if pdgid == 0:
        return 0

    # photon, electron, muon
    if pdgid in [22, 11, 13]:
        return pdgid

    # charged hadron
    if abs(charge) > 0:
        return 211

    # neutral hadron
    return 130



def get_feature_matrix(feature_dict, features):
    feats = []
    for feat in features:
        feat_arr = awkward.to_numpy(feature_dict[feat])
        feats.append(feat_arr)
    feats = np.array(feats)
    return feats.T


def process_one_file(fn, ofn):

    # output exists, do not recreate
    if os.path.isfile(ofn):
        return
    print(fn)

    fi = uproot.open(fn)


    arrs = fi['Out_Tree;1']

    # Get all branch names
    all_branches = arrs.keys()

    # Filter branches that start with 'track_' and 'cell_
    tracks_to_read = [name for name in all_branches if name.startswith("track_")]
    cells_to_read = [name for name in all_branches if name.startswith("cell_")]
    genp_to_read = [name for name in all_branches if name.startswith("particle_")]


    # Read the data for these branches
    track_data = arrs.arrays(tracks_to_read)
    cell_data = arrs.arrays(cells_to_read)
    genp_data = arrs.arrays(genp_to_read)

    ret = []

    for iev in range(arrs.num_entries):


    	#Lets extract all relevant information
    	track_features = track_to_features(track_data, iev)
    	cell_features = cell_to_features(cell_data, iev)
    	genp_features = gen_to_features(genp_data,iev)



    	# map conversion between the particle and tracks and cells
    	gp_obj_map = particle_to_obj_map(cell_data,track_data,genp_data,iev)

    	assert len(gp_obj_map) == len(genp_features["PDG"])
    	assert gp_obj_map.shape[1] == 2


    	n_tracks = len(track_features["phi"])
    	n_cells = len(cell_features["phi"])
    	n_gps = len(genp_features["PDG"])
    	print("hits={} tracks={} gps={}".format(n_cells, n_tracks, n_gps))

    	track_to_particle = {itrk: igp for igp, itrk in enumerate(gp_obj_map[:, 0]) if itrk != -1}
    	cell_to_particle = {ihit: igp for igp, ihit in enumerate(gp_obj_map[:, 1]) if ihit != -1}



    	#Making track to particle index map
    	#track_to_particle = {tri: gpi for tri, gpi in enumerate(track_data['track_parent_idx'][iev])}
    	#cell_to_particle = {ci: gpi for ci, gpi in enumerate(cell_data['cell_parent_idx'][iev])}
    	# Assuming the number of particles is known (n_particles)


    	# keep track if all genparticles were used
    	used_gps = np.zeros(n_gps, dtype=np.int64)


    	# assign all track-associated genparticles to a track
    	track_to_gp_all = assign_to_recoobj(n_tracks, track_to_particle, used_gps)
    	# assign all calohit-associated genparticles to a calohit
    	cell_to_gp_all = assign_to_recoobj(n_cells, cell_to_particle, used_gps)

    	if not np.all(used_gps == 1):
    		print("unmatched gen", genp_features["energy"][used_gps == 0])



    	gps_track = get_particle_feature_matrix(track_to_gp_all, genp_features, particle_feature_order)
    	gps_track[:, 0] = np.array([map_neutral_to_charged(map_pdgid_to_candid(p, c)) for p, c in zip(gps_track[:, 0], gps_track[:, 1])])


    	gps_cell = get_particle_feature_matrix(cell_to_gp_all, genp_features, particle_feature_order)
    	gps_cell[:, 0] = np.array([map_charged_to_neutral(map_pdgid_to_candid(p, c)) for p, c in zip(gps_cell[:, 0], gps_cell[:, 1])])
    	gps_cell[:, 1] = 0

    	assert np.all(gps_cell[:, 1] == 0)


    	X_track = get_feature_matrix(track_features, track_feature_order)
    	X_hit = get_feature_matrix(cell_features, cell_feature_order)
    	ygen_track = gps_track
    	ygen_hit = gps_cell


    	sanitize(X_track)
    	sanitize(X_hit)
    	sanitize(ygen_track)
    	sanitize(ygen_hit)

    	print("X_track={}".format(len(X_track)))
    	print("X_hit={}".format(len(X_hit)))

    	ygen_track_shape = ygen_track.shape
    	ycand_track = np.zeros(ygen_track_shape)


    	ygen_cell_shape = ygen_hit.shape
    	ycand_hit = np.zeros(ygen_cell_shape)



    	this_ev = awkward.Record({"X_track": X_track, "X_hit": X_hit,"ygen_track": ygen_track, "ygen_hit": ygen_hit,"ycand_track": ycand_track,"ycand_hit": ycand_hit})
    	ret.append(this_ev)

    ret = awkward.Record({k: awkward.from_iter([r[k] for r in ret]) for k in ret[0].fields})
    awkward.to_parquet(ret, ofn)






def process_sample(samp):
    inp = "/storage/agrp/annai/NEW_HGFlow/MLPF/outputs_cocoa_dijets/"
    outp = "/storage/agrp/annai/NEW_HGFlow/MLPF/outputs_cocoa_dijets/"

    pool = multiprocessing.Pool(8)

    inpath_samp = inp + samp
    outpath_samp = outp + samp
    infiles = list(glob.glob(inpath_samp + "/*.root"))
    if not os.path.isdir(outpath_samp):
        os.makedirs(outpath_samp)

    args = []
    for inf in infiles:
        of = inf.replace(inpath_samp, outpath_samp).replace(".root", ".parquet")
        args.append((inf, of))
    pool.starmap(process_one_file, args)


if __name__ == "__main__":
    if len(sys.argv) == 2:
        process_sample(sys.argv[1])
    if len(sys.argv) == 3:
        process_one_file(sys.argv[1], sys.argv[2])
