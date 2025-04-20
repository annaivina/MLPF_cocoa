import numpy as np
import awkward
import uproot
import vector
import glob
import os
import sys
import multiprocessing
from scipy.sparse import coo_matrix


track_feature_order = ["elemtype","pt","eta","sin_phi","cos_phi","p","chi2","ndf","radiusOfInnermostHit", "tanLambda","D0","omega","Z0","time","type",]
cluster_feature_order = ["elemtype","et","eta","sin_phi","cos_phi","energy","position.x","position.y","position.z","iTheta","energy_ecal","energy_hcal","energy_other","num_hits","sigma_x","sigma_y","sigma_z",]

# the feature matrices will be saved in this order
particle_feature_order = ["PDG", "charge", "pt", "eta", "sin_phi", "cos_phi", "energy"]

def sanitize(arr):
    arr[np.isnan(arr)] = 0.0
    arr[np.isinf(arr)] = 0.0


#Extract the Generator partciles features
def gen_to_features(genpart_data, iev):
	feats_from_gp = ['particle_pdgid','particle_pt','particle_eta','particle_phi','particle_e']
	rename_map = {
		'particle_pdgid': 'PDG',
		'particle_e': 'energy'
		}

	retg = {rename_map.get(feat, feat.split('_')[1]): genpart_data[feat][iev] for feat in feats_from_gp}

	retg["sin_phi"] = np.sin(retg["phi"])
	retg["cos_phi"] = np.cos(retg["phi"])

	retg["pt"] = retg["pt"]/1000.

	retg["energy"] = retg["energy"]/1000.

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
            -521:  -1,  521:  1, # B+-
            -431:  -1,  431:  1, # D_s+-
            -411:  -1,  411:  1, # D+-
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
	charges = [pdg_to_charge.get(pid, 0) for pid in retg['PDG']]
	retg['charge'] = charges

	unique_values = np.unique(retg['charge'])

	pdgs = np.unique(retg['PDG'])

	return awkward.Record(retg)


def track_to_features(track_data, iev):
	feats_from_track = ['track_phi','track_theta','track_qoverp','track_d0','track_z0']

	rename_map = {
		'track_z0': 'Z0',
        'track_d0': 'D0'}

	rett = {rename_map.get(feat, feat.split('_')[1]): track_data[feat][iev] for feat in feats_from_track}

	n_tr = len(rett["D0"])

	tt = np.tan(rett["theta"] / 2.0)
	eta = awkward.to_numpy(-np.log(tt, where=tt > 0))
	eta[tt <= 0] = 0.0
	rett["eta"] = eta


	rett["sin_phi"] = np.sin(rett["phi"])
	rett["cos_phi"] = np.cos(rett["phi"])

	p = 1/abs(rett["qoverp"])
	rett["p"] = p/1000.
	rett["pt"]= rett["p"] * np.sin(rett["theta"])


	new_elements = ['chi2', 'ndf','radiusOfInnermostHit','tanLambda','omega','time','type']
	for elem in new_elements:
		rett[elem] = np.zeros(n_tr, dtype=np.float32)

	# track is always type 1
	rett["elemtype"] = 1 * np.ones(n_tr, dtype=np.float32)

	return awkward.Record(rett)



def cluster_to_features(topo_data,cell_data, iev):

	feats_from_topo = ['topo_e']
	rename_map = {'topo_e': 'energy'}
	retcl = {rename_map.get(feat, feat.split('_')[1]): topo_data[feat][iev] for feat in feats_from_topo}

	retcl['energy'] = retcl['energy']/1000.

	topo_eta = topo_data['topo_bary_eta'][iev]
	topo_phi = topo_data['topo_bary_phi'][iev]
	topo_rho = topo_data['topo_bary_rho'][iev]

	theta = 2 * np.arctan(np.exp(-topo_eta))

	topo_x = topo_rho * np.sin(theta) * np.cos(topo_phi)
	topo_y = topo_rho * np.sin(theta) * np.sin(topo_phi)
	topo_z = topo_rho * np.cos(theta)

	retcl['position.x'] = topo_x
	retcl['position.y'] = topo_y
	retcl['position.z'] = topo_z

	retcl['et'] = (retcl['energy'] * np.sin(theta))

	retcl['sin_phi'] = np.sin(topo_phi)
	retcl['cos_phi'] = np.cos(topo_phi)
	retcl['eta'] = topo_eta
	retcl['iTheta'] = theta

	#Lets make sigma_x,y,z and energy_ecal,hcak
	cell_e = cell_data['cell_e'][iev]
	cell_x = cell_data['cell_x'][iev]
	cell_y = cell_data['cell_y'][iev]
	cell_z = cell_data['cell_z'][iev]
	cell_l = cell_data['cell_layer'][iev]
	#Convert! Starts from 1!!
	cell_cl_idx = awkward.to_numpy(cell_data['cell_topo_idx'][iev]) - 1
	cl_idx = awkward.to_numpy(topo_data['topo_idx'][iev])-1

	energy_ecal = -1 * np.ones(len(retcl['et']), dtype=np.float32)
	energy_hcal = -1 * np.ones(len(retcl['et']), dtype=np.float32)
	energy_oth = -1 * np.ones(len(retcl['et']), dtype=np.float32)
	sig_x = -1 * np.ones(len(retcl['et']), dtype=np.float32)
	sig_y = -1 * np.ones(len(retcl['et']), dtype=np.float32)
	sig_z = -1 * np.ones(len(retcl['et']), dtype=np.float32)
	numofcells = -1 * np.ones(len(retcl['et']), dtype=np.int32)

	for i, idx in enumerate(cl_idx):

		is_cell_from_cl = cell_cl_idx == idx

		is_in_ecal = cell_l <=2
		is_in_hcal = (cell_l >=3) & (cell_l <=5)
		is_other = cell_l > 5

		cl_e_ecal = np.sum(cell_e[is_cell_from_cl & is_in_ecal])
		cl_e_hcal = np.sum(cell_e[is_cell_from_cl & is_in_hcal])
		cl_e_oth  = np.sum(cell_e[is_cell_from_cl & is_other])

		energy_ecal[i] = cl_e_ecal/1000.
		energy_hcal[i] = cl_e_hcal/1000.
		energy_oth[i]  = cl_e_oth/1000.

		sig_x[i] = np.std(cell_x[is_cell_from_cl])
		sig_y[i] = np.std(cell_y[is_cell_from_cl])
		sig_z[i] = np.std(cell_z[is_cell_from_cl])

		numofcells[i] = np.sum(is_cell_from_cl)


	retcl['energy_ecal'] = np.array(energy_ecal)
	retcl['energy_hcal'] = np.array(energy_hcal)
	retcl['energy_other'] = np.array(energy_oth)
	retcl['sigma_x'] = sig_x
	retcl['sigma_y'] = sig_y
	retcl['sigma_z'] = sig_z
	retcl['num_hits'] = numofcells

	#cell is always type 2 (this is from their code )
	retcl["elemtype"] = 2 * np.ones(len(retcl['energy']), dtype=np.int32)

	return awkward.Record(retcl)

#From Nilotpal - topo_particle_conv
def invert_topo_part(cell_parent_list,cell_parent_energy,cell_topo_idx):

	tmp_topo_particle_idx = [];
	tmp_topo_particle_energy = []
	unique_topo_idx = np.unique(cell_topo_idx)
	for topo_idx in unique_topo_idx:
		cell_idx_in_topo = np.where(cell_topo_idx == topo_idx)[0]
		p_indices = np.hstack([np.array(x, dtype=np.int32) for x in cell_parent_list[cell_idx_in_topo]])
		p_energies = np.hstack([np.array(x, dtype=np.float32) for x in cell_parent_energy[cell_idx_in_topo]])

		if len(p_indices) == 0:
			tmp_topo_particle_idx.append(np.array([-1],dtype=np.int32))
			tmp_topo_particle_energy.append(np.array([-1],dtype=np.float32))
			continue

		max_index = p_indices.max()
		summed_energies = np.zeros(max_index + 1,dtype=np.float32)
		np.add.at(summed_energies, p_indices, p_energies)

		part_indices = np.nonzero(summed_energies)[0]
		part_energies = summed_energies[part_indices]

		tmp_topo_particle_idx.append(part_indices)
		tmp_topo_particle_energy.append(part_energies)

	topo_particle_idx = np.array(tmp_topo_particle_idx, dtype=object)
	topo_particle_energy = np.array(tmp_topo_particle_energy, dtype=object)

	return topo_particle_idx, topo_particle_energy

#Project partciles to clusters or tracks, merge partciles into one if not matched to clusters, update the info
def particle_to_obj_map(cell_data,topo_data,track_data,genp_features,iev):

	#Initiate arrays
	ngens   = len(genp_features["PDG"])
	ntracks = awkward.count(track_data["track_pdgid"][iev])
	ntopo   = awkward.count(topo_data["topo_e"][iev])
	cell_topo_idx = awkward.to_numpy(cell_data['cell_topo_idx'][iev]) - 1 #converting to start from 0
	cell_particle_list = cell_data['cell_parent_list'][iev]
	cell_particle_energy = cell_data['cell_parent_energy'][iev]
	track_to_parent_idx = track_data['track_parent_idx'][iev]
	#Number of nodes - rows in incidence matrix
	nnodes = ntopo + ntracks

	#Obtain the inverted topo to partciles associations
	topo_particle_idx,topo_particle_energy = invert_topo_part(cell_particle_list,cell_particle_energy,cell_topo_idx)

	#Create incidence matrix
	incidence_matrix = np.zeros((ngens, nnodes))


	# add tracks
	for track_idx, part_idx in enumerate(track_to_parent_idx):
		incidence_matrix[part_idx, track_idx] = 1.0


	for topo_idx, part_idxs in enumerate(topo_particle_idx):
		if(len(part_idxs)>0):
			#print(topo_particle_energy[topo_idx])
			#print(part_idx, topo_idx, ntracks, incidence_matrix.shape, topo_particle_energy.shape)
			part_idxs = part_idxs.astype(np.int32)
			incidence_matrix[part_idxs, topo_idx + ntracks] = topo_particle_energy[topo_idx]



# 		check for TC w/o associated particles
# 	if (incidence_matrix.sum(axis=0) == 0).any():
# 		noisy_cols = np.where(incidence_matrix.sum(axis=0) == 0)[0]
# 		fake_rows  = np.arange(len(noisy_cols)) + ngens
# 		incidence_matrix[fake_rows, noisy_cols] = 1.0



# 	# Initialize an array to store the best match for each particle (track or topo)
	particle_to_obj = -1 * np.ones((ngens, 2), dtype=np.int32)  # 2 columns: track, cell
	set_used_tracks = set()
	set_used_clusters = set()
	#Sort gen particles according to the energies
	psorted_e = sorted(range(ngens), key=lambda x: genp_features["energy"][x], reverse=True)
	for particle_idx in psorted_e:
		particle_row = incidence_matrix[particle_idx,:]

		if 1.0 in particle_row[:ntracks]:
			track_idx = np.argmax(particle_row[:ntracks])
			if track_idx not in set_used_tracks:
				particle_to_obj[particle_idx, 0] = track_idx
				set_used_tracks.add(track_idx)

		if particle_to_obj[particle_idx, 0] == -1:
 			cluster_associations = particle_row[ntracks:]
 			strongest_association_idx = np.argmax(cluster_associations)
 			if strongest_association_idx not in set_used_clusters:
 				particle_to_obj[particle_idx, 1] = strongest_association_idx
 				set_used_clusters.add(strongest_association_idx)

	#Check how many partciles are unmatched
	unmatched = np.where((particle_to_obj[:, 0] == -1) & (particle_to_obj[:, 1] == -1))[0]
	print("unmatched partciles: {}".format(unmatched))

	##--------Copy from the MLPF code------
	mask_gp_unmatched = np.ones(ngens,dtype=bool)
	pt_arr = np.array(awkward.to_numpy(genp_features["pt"]))
	eta_arr = np.array(awkward.to_numpy(genp_features["eta"]))
	phi_arr = np.array(awkward.to_numpy(genp_features["phi"]))
	energy_arr = np.array(awkward.to_numpy(genp_features["energy"]))

	# now merge unmatched genparticles to their closest genparticle
	gp_merges_gp0 = []
	gp_merges_gp1 = []
	for igp_unmatched in unmatched:
		mask_gp_unmatched[igp_unmatched] = False
		particle_row_un = incidence_matrix[igp_unmatched,:]
		cluster_associations_un = particle_row_un[ntracks:]
		idx_best_cluster = np.argmax(cluster_associations_un)
		idx_gp_bestcluster = np.where(particle_to_obj[:, 1] == idx_best_cluster)[0]

		if len(idx_gp_bestcluster) != 1:
			print("unmatched pt=", pt_arr[igp_unmatched])
			continue

		idx_gp_bestcluster = idx_gp_bestcluster[0]

		#Array of partciles which are having the same index and partciles which are unmatched.
		gp_merges_gp0.append(idx_gp_bestcluster)
		gp_merges_gp1.append(igp_unmatched)

		vec0 = vector.obj(
			pt = genp_features["pt"][igp_unmatched],
			eta= genp_features["eta"][igp_unmatched],
			phi= genp_features["phi"][igp_unmatched],
			e  = genp_features["energy"][igp_unmatched],
        )
		vec1 = vector.obj(
        	pt = genp_features["pt"][idx_gp_bestcluster],
        	eta= genp_features["eta"][idx_gp_bestcluster],
        	phi= genp_features["phi"][idx_gp_bestcluster],
        	e  = genp_features["energy"][idx_gp_bestcluster],
        )

		vec = vec0 + vec1
		pt_arr[idx_gp_bestcluster] = vec.pt
		eta_arr[idx_gp_bestcluster] = vec.eta
		phi_arr[idx_gp_bestcluster] = vec.phi
		energy_arr[idx_gp_bestcluster] = vec.energy


	genp_features_new = {
        	"PDG": np.abs(genp_features["PDG"][mask_gp_unmatched]),
        	"charge": genp_features["charge"][mask_gp_unmatched],
        	"pt": pt_arr[mask_gp_unmatched],
        	"eta": eta_arr[mask_gp_unmatched],
        	"sin_phi": np.sin(phi_arr[mask_gp_unmatched]),
        	"cos_phi": np.cos(phi_arr[mask_gp_unmatched]),
        	"energy": energy_arr[mask_gp_unmatched],
    }

	particle_to_obj = particle_to_obj[mask_gp_unmatched]

    #Compare number of new partciles:
	print("Number of old partciles: {}".format(len(genp_features['PDG'])))
	print("Number of new partciles: {}".format(len(genp_features_new['PDG'])))
	assert (np.sum(genp_features_new["energy"]) - np.sum(genp_features["energy"])) < 1e-2

	genp_features_new_array = awkward.Array(genp_features_new)


	return awkward.Record(genp_features_new),particle_to_obj


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


def process_one_file(file_number,fn, ofn):

    # output exists, do not recreate
    if os.path.isfile(ofn):
        return
    print(fn)

    fi = uproot.open(fn)
    arrs = fi['Out_Tree;4']
    # Get all branch names
    all_branches = arrs.keys()

    # Filter branches that start with 'track_' and 'cell_
    tracks_to_read = [name for name in all_branches if name.startswith("track_")]
    cells_to_read = [name for name in all_branches if name.startswith("cell_")]
    genp_to_read = [name for name in all_branches if name.startswith("particle_")]
    topo_to_read = [name for name in all_branches if name.startswith("topo_")]
    nodes_to_read = [name for name in all_branches if name.startswith("node_")]

    # Read the data for these branches
    track_data = arrs.arrays(tracks_to_read)
    cell_data = arrs.arrays(cells_to_read)
    genp_data = arrs.arrays(genp_to_read)
    topo_data = arrs.arrays(topo_to_read)
    node_data = arrs.arrays(nodes_to_read)

    ret = []
    file_ids = []  # To store file identifiers
    event_ids = []  # To store event numbers

    for iev in range(arrs.num_entries):

    	print(iev)
    	#Lets extract all relevant information
    	track_features = track_to_features(track_data, iev)
    	cluster_features = cluster_to_features(topo_data,cell_data,iev)
    	genp_features = gen_to_features(genp_data,iev)

    	cell_energy = cell_data['cell_e'][iev]


    	n_tracks = len(track_features["phi"])
    	n_clusters = len(cluster_features["energy"])
    	n_gps = len(genp_features["PDG"])
    	n_cells = len(cell_energy)

    	# if(n_tracks==0 or n_gps==0 or n_cells==0):
#     		print("number of tracks, gens or clusters is less then 1, skipping events")
#     		continue

    	#map conversion between the particle and tracks and cells
    	genp_features,gp_obj_map = particle_to_obj_map(cell_data,topo_data,track_data,genp_features,iev)
    	assert len(gp_obj_map) == len(genp_features["PDG"])
    	assert gp_obj_map.shape[1] == 2


    	n_tracks = len(track_features["phi"])
    	n_clusters = len(cluster_features["energy"])
    	n_gps = len(genp_features["PDG"])
    	print("clusters={} tracks={} gps={}".format(n_clusters, n_tracks, n_gps))


    	track_to_particle = {itrk: igp for igp, itrk in enumerate(gp_obj_map[:, 0]) if itrk != -1}
    	cluster_to_particle = {ihit: igp for igp, ihit in enumerate(gp_obj_map[:, 1]) if ihit != -1}

    	#keep track if all genparticles were used
    	used_gps = np.zeros(n_gps, dtype=np.int64)

    	#assign all track-associated genparticles to a track
    	track_to_gp_all = assign_to_recoobj(n_tracks, track_to_particle, used_gps)
    	#assign all cluster-associated genparticles to a calohit
    	cluster_to_gp_all = assign_to_recoobj(n_clusters, cluster_to_particle, used_gps)
    	assert np.all(used_gps == 1)


    	gps_track = get_particle_feature_matrix(track_to_gp_all, genp_features, particle_feature_order)
    	gps_track[:, 0] = np.array([map_neutral_to_charged(map_pdgid_to_candid(p, c)) for p, c in zip(gps_track[:, 0], gps_track[:, 1])])


    	gps_cluster = get_particle_feature_matrix(cluster_to_gp_all, genp_features, particle_feature_order)
    	gps_cluster[:, 0] = np.array([map_charged_to_neutral(map_pdgid_to_candid(p, c)) for p, c in zip(gps_cluster[:, 0], gps_cluster[:, 1])])
    	gps_cluster[:, 1] = 0

    	assert np.all(gps_cluster[:, 1] == 0)


    	X_track = get_feature_matrix(track_features, track_feature_order)
    	X_cluster = get_feature_matrix(cluster_features, cluster_feature_order)
    	ygen_track = gps_track
    	ygen_cluster = gps_cluster

    	sanitize(X_track)
    	sanitize(X_cluster)
    	sanitize(ygen_track)
    	sanitize(ygen_cluster)

		#Substituting the Cands with zeros
    	ygen_track_shape = ygen_track.shape
    	ycand_track = np.zeros(ygen_track_shape)

    	ygen_cluster_shape = ygen_cluster.shape
    	ycand_cluster = np.zeros(ygen_cluster_shape)

    	this_ev = awkward.Record({"X_track": X_track,
    							  "X_cluster": X_cluster,
    							  "ygen_track": ygen_track,
    							  "ygen_cluster": ygen_cluster,
    							  "ycand_track": ycand_track,
    							  "ycand_cluster": ycand_cluster,
    							  "file_id": int(file_number),
    							  "event_id": int(iev)})
    	ret.append(this_ev)

    ret = awkward.Record({k: awkward.from_iter([r[k] for r in ret]) for k in ret[0].fields})
    awkward.to_parquet(ret, ofn)




def process_sample(samp):
    inp = "/Users/annushka/Desktop/MLPF/MLPF_produce/cocoa_dijet_outputs/"
    outp = "/Users/annushka/Desktop/MLPF/MLPF_produce/cocoa_dijet_outputs/"

    pool = multiprocessing.Pool(8)

    inpath_samp = inp + samp
    outpath_samp = outp + samp
    infiles = list(glob.glob(inpath_samp + "/*.root"))
    if not os.path.isdir(outpath_samp):
        os.makedirs(outpath_samp)

    args = []
    for inf in infiles:
        of = inf.replace(inpath_samp, outpath_samp).replace(".root", ".parquet")
        file_number = os.path.basename(inf).split('_')[-1].replace(".root", "")
        args.append((file_number,inf, of))
    pool.starmap(process_one_file, args)


if __name__ == "__main__":
    if len(sys.argv) == 2:
        process_sample(sys.argv[1])
    if len(sys.argv) == 4:
        process_one_file(sys.argv[1],sys.argv[2], sys.argv[3])
