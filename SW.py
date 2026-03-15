import os
import glob
import numpy as np
from scipy.stats import pearsonr  #import pearson corr

ts_dir = "./ica_timeseries"          # time series generated in the dual regression step
out_dir = "./dfc_sliding_window"     
n_comp = 20                   # number of ica components
tr = 2.2                         # TR refers to the time resolution of the fmri scan change accordingly to the data    
window_sec = 40.0               # widnow size 
overlap = 0.90                     # overlap between widnows
window_size = int(window_sec / tr) # define widnow size in trs : note when thsi is done for 40s and tr of 2.2 the widnow size is exactly 39.6s
step_size = int(window_size * (1 - overlap))
print(f"window size in trs : {window_size}")
print(f"step size in trs: {step_size}")
os.makedirs(out_dir, exist_ok=True)




def sw_dfc(ts, window_size, step_size):  # define the widnow as a function to be applied
    T, N = ts.shape
    windows = []
    for start in range(0, T - window_size + 1, step_size):
        segment = ts[start:start + window_size]
        # Pearson correlation matrix
        corr = np.corrcoef(segment, rowvar=False)
        windows.append(corr)
    return np.array(windows)


files = sorted(glob.glob(os.path.join(ts_dir, "*_ICA20_timeseries.npy")))
print(f"Found {len(files)} subjects")

# run the sliding window across all idnividuals  and stre them in the output dir 
for f in files:
    subj = os.path.basename(f).replace("_ICA20_timeseries.npy", "")
    print(f"sub id {subj}")
    ts = np.load(f)   # shape: (T x 20)
    if ts.shape[1] != n_comp:
        raise ValueError(f"{subj}: should be {n_comp} components, but is  {ts.shape[1]}")
    dfc = sw_dfc(ts, window_size, step_size)
    out_file = os.path.join(out_dir, f"{subj}_dfc_sliding.npy")
    np.save(out_file, dfc)
    print(f" dfc_shape{dfc.shape}")

print("finished")