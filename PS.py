import os
import glob
import numpy as np
from scipy.signal import butter, filtfilt, hilbert   # import hilbert transofr

ts_dir = "./ica_timeseries"     #directorly of time series 
out_dir = "./dfc_phase_synchrony"   # output directory 
n_comp = 20
tr = 2.2                        # Tr change according to fmri temporal resolution
freq = 0.05                     # Freq targeted by the bandpass this one is consistaant with the thesis result however frequencies between 0.04-0.07 should work 
bandwidth = 0.02                # +/- range of bandwidth
window_sec = 60.0               # pvl widnow size 
overlap = 0.90                  # overlap

#extra params
fs = 1.0 / tr
window_size = int(window_sec / tr)
step_size = int(window_size * (1 - overlap))

low = freq - bandwidth
high = freq + bandwidth

print(f"Window size: {window_size} TRs")
print(f"Step size: {step_size} TRs")
print(f"Band-pass: {low:.3f}–{high:.3f} Hz")

os.makedirs(out_dir, exist_ok=True)

#bandpass filter 
def bandpass(ts, low, high, fs, order=4):
    nyq = 0.5 * fs
    b, a = butter(order, [low / nyq, high / nyq], btype="band")
    return filtfilt(b, a, ts, axis=0)

# PVL window applied as a method
def window_plv(phase, window_size, step_size):

    T, N = phase.shape
    windows = []

    for start in range(0, T - window_size + 1, step_size):
        seg = phase[start:start + window_size]

        plv = np.zeros((N, N))
        for i in range(N):
            for j in range(i, N):
                diff = seg[:, i] - seg[:, j]    # compute phase difference before applying the PVL
                val = np.abs(np.mean(np.exp(1j * diff)))
                plv[i, j] = plv[j, i] = val

        windows.append(plv)

    return np.array(windows)

#main loop 
files = sorted(glob.glob(os.path.join(ts_dir, "*_ICA20_timeseries.npy")))
print(f"Found {len(files)} subjects")

for f in files:
    subj = os.path.basename(f).replace("_ICA20_timeseries.npy", "")
    print(f"Processing {subj}")

    ts = np.load(f)  # (T x 20)

    if ts.shape[1] != n_comp:
        raise ValueError(f"{subj}: wrong number of components")
    ts_filt = bandpass(ts, low, high, fs)
    phase = np.angle(hilbert(ts_filt, axis=0))  # extract the phase only from the 

    plv = window_plv(phase, window_size, step_size)

    out_file = os.path.join(out_dir, f"{subj}_plv_sliding.npy")
    np.save(out_file, plv)
    print(f"  Saved PLV shape: {plv.shape}")

print("All subjects processed.")