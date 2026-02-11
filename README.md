# Learning Alzheimer's Disease Signatures in EEG  
## Spiking Neural Networks and E/I Imbalance Simulations

![Alt text](graphabs3.png)

This repository accompanies the paper:

**“Learning Alzheimer's Disease Signatures in EEG: Spiking Networks and E/I Imbalance Simulations”** Mamon, Talanov and Crimi

The project presents a unified neuromorphic and mechanistic framework for studying Alzheimer’s disease (AD) from resting-state EEG data by combining:

- Spiking Neural Network (SNN)–based EEG classification  
- Network-based statistics (NBS) on functional connectivity  
- Biophysically grounded spiking network simulations of excitation–inhibition (E/I) imbalance  
- Functional connectivity–informed large-scale simulations  

The goal is to link **data-driven classification** with **mechanistic modeling**, providing interpretable EEG biomarkers grounded in cortical circuit dynamics.

---

## Repository Structure

- `classification/`: Contains Jupyter notebooks for loading data, training models, and validation.
- `connectivity/`: Scripts for functional connectivity analysis (PLV) and network-based statistics.
- `simulations/`: Scripts for E/I imbalance simulations using NEST.
- `others/`: Visualization tools and utilities.
- `train_snn.py`: Main script for training SNN models using snnTorch.
- `validate_on_simulation.py`: Validation of SNN models in simulation before hardware deployment.
- `deploy_spinnaker.py`: Deployment script for SpiNNaker neuromorphic hardware.

## Installation

### Prerequisites

Ensure you have Python 3.8+ installed. You will need the following libraries:

```bash
pip install numpy pandas scikit-learn torch snntorch imbalanced-learn joblib mne scipy matplotlib networkx
```

For simulations involving NEST:
```bash
pip install nest-simulator
```

For SpiNNaker deployment:
```bash
pip install sPyNNaker
```
(See [sPyNNaker documentation](http://spinnakermanchester.github.io/latest/spynnaker_install.html) for detailed installation instructions).

## Usage

### 1. SNN Training

Train the Spiking Neural Network classifier on your **pre-extracted EEG features**.

**Input Data Requirements:**
- The script expects labeled feature data, **not raw EEG time-series**.
- **Features (`data/X_ml.csv`)**: A CSV file where each row represents an epoch/window and columns represent extracted features (e.g., spectral power bands, complexity measures, PLV).
- **Labels (`data/y_ml.csv`)**: A CSV file containing the class labels (e.g., 'AD', 'HC3') corresponding to each row in the feature file.

```bash
python train_snn.py
```
This script will:
- Load and preprocess feature data (SMOTE oversampling, scaling).
- Train an SNN model using `snnTorch`.
- Save the trained model parameters and scaler.
- Convert the model for SpiNNaker compatibility.

### 2. Validation

Validate the converted model parameters in a PyTorch-based simulation to ensure accuracy before deploying to hardware.

```bash
python validate_on_simulation.py
```

### 3. SpiNNaker Deployment

Deploy the validated SNN model onto SpiNNaker hardware or the SpiNNaker software simulator.

```bash
python deploy_spinnaker.py
```

### 4. Functional Connectivity Analysis

Scripts in `connectivity/` allow you to compute Phase Locking Value (PLV) and perform Network-Based Statistics (NBS).

**Input Data Requirements:**
- These scripts expect **raw or preprocessed EEG data files** in EEGLAB format (`.set`).
- The data should be organized in a BIDS-like structure (e.g., `sub-001/eeg/sub-001_task-rest_eeg.set`).

Example: Compute PLV for EEG data.
```bash
python connectivity/compute_PLV.py
```

### 5. E/I Imbalance Simulations

Run biophysical simulations of cortical circuits with varying Excitation/Inhibition ratios to model AD progression.

```bash
python simulations/EI_decay.py
```
This will generate power spectra plots comparing different E/I conditions (AD, MCI, HC).

### 6. Visualization

Visualize spiking network activity dynamically.

```bash
python others/dyn_vis_snn.py
```

## Citation

If you use this code or data, please cite our paper:

> Mamon, Talanov and Crimi. "Learning Alzheimer's Disease Signatures in EEG: Spiking Networks and E/I Imbalance Simulations". [Journal/Conference Name], [Year].

---
*Note: Code outside the main folders is work in progress.*
