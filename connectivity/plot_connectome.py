import pandas as pd
from nilearn import plotting
import numpy as np
import mne 
 
# Load standard 10–20 montage
montage = mne.channels.make_standard_montage("standard_1020")
# Get channel positions (in meters)
pos = montage.get_positions()
ch_pos = pos["ch_pos"]   # dict: channel -> (x, y, z)

# Desired electrode order
electrodes = [
    'Fp1','Fp2','F3','F4','C3','C4','P3','P4','O1','O2',
    'F7','F8','Fz','Cz','Pz'
]
# Build (N, 3) array in the specified order and scale
coords = np.array([ch_pos[ch] for ch in electrodes])*800

#connectome = pd.read_csv("Conn_mat.txt", header=None)
connectome = np.load("NBS_alpha_left_component0.npy")

plotting.view_connectome(connectome, coords, linewidth=5.0, node_size=5.0)
