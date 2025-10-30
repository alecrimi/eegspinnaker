#Voytek, Bradley, et al. "Age-related changes in 1/f neural electrophysiological noise." 
#Journal of neuroscience 35.38 (2015): 13257-13265.

# Martínez‐Cañada, Pablo, et al. "Combining aperiodic 1/f slopes and brain simulation: An EEG/MEG proxy marker of excitation/inhibition imbalance in Alzheimer's disease." 
# Alzheimer's & Dementia: Diagnosis, Assessment & Disease Monitoring 15.3 (2023): e12477.

import mne
import numpy as np
import matplotlib.pyplot as plt
from fooof import FOOOF

# -----------------------------
# Step 1: Load your EEG data
# -----------------------------
# Example: a raw EEG file in EDF format
raw = mne.io.read_raw_edf('AD_subject.edf', preload=True)

# Pick EEG channels (exclude EOG, EMG)
raw.pick_types(eeg=True)

# Filter data: 1-40 Hz
raw.filter(1., 40., fir_design='firwin')

# -----------------------------
# Step 2: Compute PSD (Welch method)
# -----------------------------
psds, freqs = mne.time_frequency.psd_welch(raw, fmin=1, fmax=40, n_fft=2048)

# Average across channels
psd_mean = psds.mean(axis=0)

# -----------------------------
# Step 3: Fit 1/f slope with FOOOF
# -----------------------------
fm = FOOOF(peak_width_limits=[1, 8], max_n_peaks=6, min_peak_height=0.1, verbose=False)
fm.fit(freqs, psd_mean)

# Extract slope (aperiodic exponent)
slope = fm.aperiodic_params_[1]
offset = fm.aperiodic_params_[0]

print(f"1/f slope (aperiodic exponent) = {slope:.2f}")

# -----------------------------
# Step 4: Plot PSD and fit
# -----------------------------
fm.plot()
plt.title('EEG PSD with 1/f fit')
plt.show()
