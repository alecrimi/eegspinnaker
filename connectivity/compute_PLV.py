import os
import mne
import numpy as np
from scipy.signal import hilbert, butter, sosfiltfilt

# --------- CONFIGURATION ---------
bids_root = "./"  # root folder of your BIDS dataset
channels_19 = ['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2', 'F7', 'F8', 'T3', 'T4', 'T5', 'T6', 'Fz', 'Cz', 'Pz']  # standard 10-20 19 channels




# Define frequency bands
BANDS = {
#    'delta': (0.5,4),
#    'theta': (4, 8),
    'alpha': (8, 13)#,
#    'beta': (13, 30),
#    'gamma': (30, 45)
}

# Output folders (at root level)
OUTPUT_FOLDERS = {
    'delta': os.path.join(bids_root, 'connectivity_delta'),
    'theta': os.path.join(bids_root, 'connectivity_theta'),
    'alpha': os.path.join(bids_root, 'connectivity_alpha'),
    'beta': os.path.join(bids_root, 'connectivity_beta'),
    'gamma': os.path.join(bids_root, 'connectivity_gamma')
}

# Create output folders
for folder in OUTPUT_FOLDERS.values():
    os.makedirs(folder, exist_ok=True)

# ---------------------------------

def bandpass_filter(data, lowcut, highcut, fs, order=4):
    """
    Bandpass filter the data.
    
    Parameters:
    -----------
    data : array (n_channels, n_samples)
    lowcut : float - Low frequency cutoff
    highcut : float - High frequency cutoff
    fs : float - Sampling frequency
    order : int - Filter order
    
    Returns:
    --------
    filtered_data : array (n_channels, n_samples)
    """
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    
    # Check if frequencies are valid
    if low <= 0 or high >= 1:
        raise ValueError(f"Invalid frequency range [{lowcut}, {highcut}] Hz for fs={fs} Hz")
    
    sos = butter(order, [low, high], btype='band', output='sos')
    filtered = np.zeros_like(data)
    
    for ch in range(data.shape[0]):
        filtered[ch, :] = sosfiltfilt(sos, data[ch, :])
    
    return filtered


def compute_plv(data, lowcut, highcut, fs):
    """
    Compute Phase Locking Value for a specific frequency band.
    
    Parameters:
    -----------
    data : array (n_channels, n_samples)
    lowcut : float - Low frequency cutoff
    highcut : float - High frequency cutoff
    fs : float - Sampling frequency
    
    Returns:
    --------
    plv_matrix : array (n_channels, n_channels)
    """
    # Bandpass filter
    filtered = bandpass_filter(data, lowcut, highcut, fs)
    
    # Compute analytic signal and extract phase
    analytic_signal = hilbert(filtered, axis=-1)
    phases = np.angle(analytic_signal)
    
    # Compute PLV between all pairs
    n_channels = phases.shape[0]
    plv_matrix = np.zeros((n_channels, n_channels))
    
    for i in range(n_channels):
        for j in range(n_channels):
            phase_diff = phases[i, :] - phases[j, :]
            plv_matrix[i, j] = np.abs(np.mean(np.exp(1j * phase_diff)))
    
    return plv_matrix


# Dictionary to store PLV matrices for each band across all subjects
all_plvs = {band: [] for band in BANDS.keys()}

# List of subjects
subjects = [d for d in os.listdir(bids_root) if d.startswith('sub-')]

print("="*70)
print(f"PROCESSING {len(subjects)} SUBJECTS")
print("="*70)
print(f"Frequency bands: {BANDS}")
print(f"\nOutput folders:")
for band, folder in OUTPUT_FOLDERS.items():
    print(f"  {band:6s}: {folder}")
print("="*70)

for sub_idx, sub in enumerate(subjects, 1):
    eeg_folder = os.path.join(bids_root, sub, 'eeg')
    if not os.path.exists(eeg_folder):
        continue
    
    print(f"\n[{sub_idx}/{len(subjects)}] Processing {sub}...")
    
    for file in os.listdir(eeg_folder):
        if file.endswith('.set'):
            file_path = os.path.join(eeg_folder, file)
            
            try:
                # Load EEG data
                raw = mne.io.read_raw_eeglab(file_path, preload=True, verbose=False)
                
                # Pick 19 channels
                available_channels = [ch for ch in channels_19 if ch in raw.ch_names]
                if len(available_channels) < len(channels_19):
                    print(f"  Warning: Only {len(available_channels)}/{len(channels_19)} channels found")
                
                raw.pick_channels(available_channels)
                
                # Get data array and sampling frequency
                data = raw.get_data()  # shape: (n_channels, n_times)
                fs = raw.info['sfreq']
                
                print(f"  Data: {data.shape[0]} channels, {data.shape[1]} samples, fs={fs} Hz")
                
                # Process each frequency band
                for band_name, (lowcut, highcut) in BANDS.items():
                    try:
                        # Compute PLV for this band
                        plv_matrix = compute_plv(data, lowcut, highcut, fs)
                        
                        # Save PLV matrix in band-specific folder at root level
                        # Use subject name in filename for easy identification
                        out_file = os.path.join(OUTPUT_FOLDERS[band_name], 
                                               f"{sub}_{band_name}_plv.npy")
                        np.save(out_file, plv_matrix)
                        
                        # Store for averaging
                        all_plvs[band_name].append(plv_matrix)
                        
                        print(f"    {band_name} ({lowcut}-{highcut} Hz): ✓ Saved to {out_file}")
                        
                    except Exception as e:
                        print(f"    {band_name} ({lowcut}-{highcut} Hz): ✗ Error: {e}")
                
            except Exception as e:
                print(f"  Error loading {file}: {e}")

# Compute average PLV for each band across all subjects
print("\n" + "="*70)
print("COMPUTING AVERAGE PLV MATRICES PER BAND")
print("="*70)

for band_name, plv_list in all_plvs.items():
    if plv_list:
        # Compute mean across all subjects
        avg_plv = np.mean(np.stack(plv_list), axis=0)
        
        # Save average PLV in the band folder
        avg_file = os.path.join(OUTPUT_FOLDERS[band_name], f"average_{band_name}_plv.npy")
        np.save(avg_file, avg_plv)
        
        print(f"{band_name:6s}: {len(plv_list)} subjects, mean PLV = {avg_plv.mean():.4f}")
        print(f"         Average saved to: {avg_file}")
    else:
        print(f"{band_name:6s}: No PLV matrices computed")

print("\n" + "="*70)
print("PROCESSING COMPLETE")
print("="*70) 
