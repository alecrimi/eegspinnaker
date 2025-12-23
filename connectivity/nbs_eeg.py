import os
import numpy as np
import bct

# ============================================================
# SETTINGS
# ============================================================
DATA_ROOT = "./"     # contains HC/ and AD/
PRIMARY_T = 2
N_PERM = 5000
ALPHA = 0.05

# ============================================================
# LOAD DATA
# ============================================================
def load_group(group):
    mats = []
    base = os.path.join(DATA_ROOT, group)
    for sub in sorted(os.listdir(base)):
        eeg_dir = os.path.join(base, sub, "eeg", "connectivity")
        if not os.path.isdir(eeg_dir):
            continue
        files = [f for f in os.listdir(eeg_dir) if f.endswith(".npy")]
        if not files:
            continue
        mat = np.load(os.path.join(eeg_dir, files[0]))
        np.fill_diagonal(mat, 0)
        mats.append(mat)
    return np.asarray(mats)

HC = load_group("HC")
AD = load_group("AD")
assert HC.shape[1:] == AD.shape[1:], "Matrix size mismatch"
n_HC = HC.shape[0]
n_AD = AD.shape[0]
print(f"Loaded {n_HC} HC and {n_AD} AD subjects")

# ============================================================
# PREPARE DATA FOR bct.nbs_bct
# ============================================================
# bct.nbs_bct expects TWO separate arrays: (N, N, subjects_group1) and (N, N, subjects_group2)
HC_transposed = np.transpose(HC, (1, 2, 0))  # Shape: (N, N, n_HC)
AD_transposed = np.transpose(AD, (1, 2, 0))  # Shape: (N, N, n_AD)

# ============================================================
# RUN NBS (bctpy)
# ============================================================
print(f"\nRunning NBS with threshold={PRIMARY_T}, permutations={N_PERM}...")

pvals, adj, _ = bct.nbs_bct(
    HC_transposed,  # Group 1: (N, N, n_HC)
    AD_transposed,  # Group 2: (N, N, n_AD)
    PRIMARY_T,      # primary threshold
    k=N_PERM,       # permutations
    tail='right',    # tail: 'both', 'left', or 'right'
    paired=False,   # paired test
    verbose=False   # verbosity
)

# ============================================================
# RESULTS
# ============================================================
print("\nNBS results:")
sig = pvals < ALPHA
if not sig.any():
    print("No significant components found.")
else:
    print(f"Found {sig.sum()} significant component(s)")
    for i, p in enumerate(pvals):
        if p < ALPHA:
            print(f"Component {i}: p = {p:.4f}")
    
    # Adjacency matrix of first significant component
    adj_sig = adj[:, :, sig][:, :, 0]
    np.save("NBS_significant_network.npy", adj_sig)
    print("Saved NBS_significant_network.npy")
