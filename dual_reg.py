
# this is the fierst stage of dual regression since we dont need indiviuydla spatila maps for dfc consturction it is enough to run only the first stage
import os, glob
import numpy as np
import nibabel as nib

data_dir = "/mnt/c/Users/Gjirafa/Documents/python/Bold_data/full_data" # this is the idnividual data file
s_file = "group_spatial_maps_20.npy" # the already preprocessed ica group maps
out_dir = "./ica_timeseries"
os.makedirs(out_dir, exist_ok=True)
spat_map = np.load(s_file)
pinv_sgr = np.linalg.pinv(spat_map)

files = sorted(glob.glob(os.path.join(
    data_dir, "CASCA*_RS_denoised_tempfiltered_warped2std.nii*"  #keep the naming consistent
)))

for i in files:
    subject = os.path.basename(i).split("_")[0]
    print("curr_sub", subject)
    x_mat = nib.load(i).get_fdata()
    x_mat = x_mat.reshape(-1, x_mat.shape[-1]).T
    x_mat -= x_mat.mean(axis=0)
    a_mat = x_mat @ pinv_sgr
    # store each individuals ica_comp x time_series as unique numpy files make the later steps fastter
    np.save(os.path.join(out_dir, f"{subject}_ICA20_timeseries.npy"), a_mat) 
    del x_mat, a_mat

print("finishde ")