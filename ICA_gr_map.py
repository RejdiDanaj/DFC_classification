#!/usr/bin/env python3
import os
import glob
import numpy as np
import nibabel as nib
from sklearn.decomposition import PCA, FastICA



data_folder = "/mnt/c/Users/Gjirafa/Documents/python/Bold_data" # folder with stored data
n_components = 20 # number of components
memmap_file = "group_timeseries_memmap.dat" 
spatmap_out = "group_spatmap_20.npy" # group spatial maps output file name 

#load all subjects 

fmri_files = sorted(glob.glob(os.path.join(
    data_folder,
    "CASCA*_RS_denoised_tempfiltered_warped2std.nii*" # change this to the files naming convension
)))

if not fmri_files:
    raise RuntimeError("No CASCA files found in the data folder.")


total_timepoints = 0
for fmri_path in fmri_files:
    img = nib.load(fmri_path)

    total_timepoints += img.shape[-1]
    n_voxels = np.prod(img.shape[:-1])


print("Creating disk-backed memmap with shape:", (total_timepoints, n_voxels))

group_timeseries = np.memmap(
    memmap_file,
    dtype="float32",
    mode="w+",
    shape=(total_timepoints, n_voxels)
)


#load subjects in memory 

current_row = 0
n_subjects = len(fmri_files)

print("Loading subjects into memmap:")

for subject_index, fmri_path in enumerate(fmri_files, start=1):
    fmri_data = nib.load(fmri_path).get_fdata().astype("float32")
    fmri_matrix = fmri_data.reshape(-1, fmri_data.shape[-1]).T
    fmri_matrix -= fmri_matrix.mean(axis=0)
    group_timeseries[current_row:current_row + fmri_matrix.shape[0]] = fmri_matrix
    current_row += fmri_matrix.shape[0]
    progress = subject_index / n_subjects * 100
    print(f"[{subject_index}/{n_subjects}] Loaded {os.path.basename(fmri_path)} ({progress:.1f}%)")


# Step one is to run standart PCA procedure

print("Running pca across the entire group...")

pca_model = PCA(
    n_components=n_components,
    svd_solver="randomized"
)

group_pca_timeseries = pca_model.fit_transform(group_timeseries)


# Run ICA make sure to adjust iterations
print("Running ica...")

ica_model = FastICA(
    n_components=n_components,
    whiten=False,
    max_iter=1000,
    tol=1e-4,
    random_state=0
)

ica_model.fit(group_pca_timeseries)

#get group spatial maps form the ICA 
group_spatmap = ica_model.components_ @ pca_model.components_

np.save(spatmap_out, group_spatmap)

# make sure to remove the initialzed veriables in memory they take too much space and python doesnt clean them unless the kernel is restarted
del group_timeseries
del group_pca_timeseries

print("Saved group spatmap:", spatmap_out)