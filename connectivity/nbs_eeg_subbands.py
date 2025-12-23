import os
import numpy as np
import bct

# ============================================================
# SETTINGS
# ============================================================
DATA_ROOT = "./"     # contains HC/ and AD/
PRIMARY_T = 2.5
N_PERM = 5000
ALPHA = 0.01

# Define which bands to analyze
BANDS = ['theta', 'alpha', 'beta', 'gamma']

# ============================================================
# LOAD DATA FROM BAND-SPECIFIC FOLDERS
# ============================================================
def load_group_from_band(group, band):
    """
    Load all .npy files for a specific group and band.
    
    Parameters:
    -----------
    group : str
        Group name ('HC' or 'AD')
    band : str
        Band name ('theta', 'alpha', 'beta', 'gamma')
    
    Returns:
    --------
    mats : array (n_subjects, n_channels, n_channels)
    """
    mats = []
    band_folder = os.path.join(DATA_ROOT, group, f'connectivity_{band}')
    
    if not os.path.isdir(band_folder):
        print(f"Warning: Folder {band_folder} does not exist!")
        return np.array([])
    
    # Get all .npy files that are subject files (start with 'sub-')
    # Exclude: average files, any file not starting with 'sub-'
    files = [f for f in sorted(os.listdir(band_folder)) 
             if f.endswith('.npy') and f.startswith('sub-')]
    
    print(f"  {group}: Found {len(files)} subject files in {band_folder}")
    
    # Debug: show first few files
    if files:
        print(f"       First 3 files: {files[:3]}")
    
    for file in files:
        file_path = os.path.join(band_folder, file)
        mat = np.load(file_path)
        np.fill_diagonal(mat, 0)
        mats.append(mat)
    
    return np.asarray(mats)


# ============================================================
# RUN NBS FOR EACH BAND
# ============================================================
print("="*70)
print("NETWORK-BASED STATISTIC (NBS) - BAND-SPECIFIC ANALYSIS")
print("="*70)
print(f"Settings:")
print(f"  Data root: {DATA_ROOT}")
print(f"  Primary threshold: {PRIMARY_T}")
print(f"  Permutations: {N_PERM}")
print(f"  Alpha: {ALPHA}")
print("="*70)

all_results = {}

for band in BANDS:
    print(f"\n{'='*70}")
    print(f"ANALYZING {band.upper()} BAND")
    print(f"{'='*70}")
    
    # Load HC and AD data for this band
    print("Loading data...")
    HC = load_group_from_band('HC', band)
    AD = load_group_from_band('AD', band)
    
    if len(HC) == 0 or len(AD) == 0:
        print(f"ERROR: Could not load data. HC: {len(HC)}, AD: {len(AD)}")
        print("Skipping this band.")
        continue
    
    # Verify dimensions
    try:
        assert HC.shape[1:] == AD.shape[1:], "Matrix size mismatch"
    except AssertionError:
        print(f"ERROR: Matrix size mismatch - HC: {HC.shape}, AD: {AD.shape}")
        continue
    
    n_HC = HC.shape[0]
    n_AD = AD.shape[0]
    print(f"\nLoaded {n_HC} HC and {n_AD} AD subjects")
    print(f"Matrix size: {HC.shape[1]} x {HC.shape[2]}")
    
    # Prepare data for NBS
    HC_transposed = np.transpose(HC, (1, 2, 0))  # Shape: (N, N, n_HC)
    AD_transposed = np.transpose(AD, (1, 2, 0))  # Shape: (N, N, n_AD)
    
    # Run NBS for both tails
    for tail_type in ['left', 'right']:
        print(f"\n--- Testing {tail_type} tail (", end="")
        if tail_type == 'left':
            print("HC > AD, decreased in AD) ---")
        else:
            print("AD > HC, increased in AD) ---")
        
        print(f"Running {N_PERM} permutations (this may take a few minutes)...")
        
        try:
            pvals, adj, _ = bct.nbs_bct(
                HC_transposed,
                AD_transposed,
                PRIMARY_T,
                k=N_PERM,
                tail=tail_type,
                paired=False,
                verbose=False  # ← Set to False to hide permutation progress
            )
            
            sig = pvals < ALPHA
            
            if not sig.any():
                min_p = pvals.min() if len(pvals) > 0 else 1.0
                print(f"✗ No significant components (min p = {min_p:.4f})")
            else:
                print(f"✓ Found {sig.sum()} significant component(s)!")
                
                # Store results
                result_key = f"{band}_{tail_type}"
                all_results[result_key] = {
                    'pvals': pvals,
                    'adj': adj,
                    'sig_indices': np.where(sig)[0],
                    'n_HC': n_HC,
                    'n_AD': n_AD
                }
                
                # Handle adjacency matrix extraction based on shape
                # adj can be either (N, N) for single component or (N, N, n_components)
                if adj.ndim == 2:
                    # Single component case
                    if sig[0]:  # If the single component is significant
                        n_edges = np.sum(adj > 0)
                        print(f"  Component 0: p = {pvals[0]:.4f}, {n_edges} edges")
                        output_file = f"NBS_{band}_{tail_type}_component0.npy"
                        np.save(output_file, adj)
                        print(f"  Saved: {output_file}")
                elif adj.ndim == 3:
                    # Multiple components case
                    for i, p in enumerate(pvals):
                        if p < ALPHA:
                            adj_comp = adj[:, :, i]
                            n_edges = np.sum(adj_comp > 0)
                            print(f"  Component {i}: p = {p:.4f}, {n_edges} edges")
                            output_file = f"NBS_{band}_{tail_type}_component{i}.npy"
                            np.save(output_file, adj_comp)
                            print(f"  Saved: {output_file}")
                else:
                    print(f"  Unexpected adj shape: {adj.shape}")
        
        except Exception as e:
            print(f"✗ Error running NBS: {e}")

# ============================================================
# SUMMARY
# ============================================================
print("\n" + "="*70)
print("SUMMARY OF RESULTS")
print("="*70)

if all_results:
    print(f"\n{'Band':<10} {'Tail':<10} {'Components':<12} {'Subjects':<18} {'Min p-value':<15}")
    print("-"*70)
    for key, result in all_results.items():
        band, tail = key.rsplit('_', 1)
        n_sig = len(result['sig_indices'])
        min_p = result['pvals'][result['sig_indices']].min() if n_sig > 0 else 1.0
        n_subj = f"{result['n_HC']}HC/{result['n_AD']}AD"
        print(f"{band:<10} {tail:<10} {n_sig:<12} {n_subj:<18} {min_p:<15.4f}")
    
    print("\n" + "="*70)
    print("FILES SAVED:")
    print("="*70)
    saved_files = [f for f in os.listdir('.') if f.startswith('NBS_') and f.endswith('.npy')]
    if saved_files:
        for f in sorted(saved_files):
            mat = np.load(f)
            n_edges = np.sum(mat > 0)
            print(f"  {f:<50} ({mat.shape[0]}×{mat.shape[1]}, {n_edges} edges)")
    else:
        print("  No files saved (no significant results)")
        
    print("\n" + "="*70)
    print("INTERPRETATION GUIDE:")
    print("="*70)
    print("Theta band (4-8 Hz):")
    print("  • RIGHT tail: Increased connectivity in AD → pathological slowing")
    print("  • LEFT tail: Decreased connectivity in AD → rare in theta")
    print("\nAlpha band (8-13 Hz):")
    print("  • LEFT tail: Decreased connectivity in AD → typical finding (loss of alpha)")
    print("  • RIGHT tail: Increased connectivity in AD → unusual")
    print("\nBeta band (13-30 Hz):")
    print("  • LEFT tail: Decreased connectivity in AD → typical (reduced cognition)")
    print("  • RIGHT tail: Increased connectivity in AD → possible compensation")
    print("\nGamma band (30-45 Hz):")
    print("  • Variable effects, less studied in AD literature")
else:
    print("No significant results found in any band.")
    print("\nPossible reasons:")
    print("  1. PRIMARY_T too high → try lowering to 2.0 or 1.5")
    print("  2. Sample size → may need more subjects")
    print("  3. Connectivity measure → consider other metrics (coherence, PLI)")
    print("  4. Preprocessing → check filtering, artifact removal")

print("\n" + "="*70)
print("NEXT STEPS:")
print("="*70)
print("1. Visualize significant networks using network visualization tools")
print("2. Identify which brain regions are involved (map channels to ROIs)")
print("3. Compute graph metrics on significant components")
print("4. Correlate network features with clinical scores (if available)")
print("="*70)
