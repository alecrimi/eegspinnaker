# ============================================================
# PyNEST – Improved E/I imbalance with FC-based connectivity
# 10 repeats, mean ± std
# ============================================================

import nest
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch, savgol_filter
from scipy.ndimage import gaussian_filter1d
import os

# -------------------- Helper functions --------------------

def load_connectivity_matrix(group, band, data_root="./"):
    conn_file = os.path.join(data_root, group, f'connectivity_{band}', f'average_{band}_plv.npy')
    if not os.path.exists(conn_file):
        raise FileNotFoundError(f"Connectivity file not found: {conn_file}")
    conn_matrix = np.load(conn_file)
    np.fill_diagonal(conn_matrix, 0)
    print(f"    Loaded {band} FC: range [{conn_matrix.min():.3f}, {conn_matrix.max():.3f}]")
    return conn_matrix

def select_regions(conn_matrix, n_regions=19):
    node_strength = np.sum(np.abs(conn_matrix), axis=1)
    selected_indices = np.argsort(node_strength)[-n_regions:]
    region_conn = conn_matrix[np.ix_(selected_indices, selected_indices)]
    return selected_indices, region_conn

def run_simulation_with_fc(condition, band_name, g_ratio, conn_matrix, 
                           N_per_region=52, n_regions=19, frac_exc=0.8,
                           p_conn_local=0.2, coupling_strength=1.5,
                           nu_ext=15.0, sim_time=10000.0, warmup=3000.0,
                           seed=42, record_fraction=0.2, smooth_spectrum=True):

    nest.ResetKernel()
    nest.resolution = 0.1
    nest.set_verbosity("M_WARNING")
    rng_seed = seed + hash(condition + band_name) % 1000
    np.random.seed(rng_seed)
    nest.SetKernelStatus({"rng_seed": rng_seed})

    N_E_region = int(N_per_region * frac_exc)
    N_I_region = N_per_region - N_E_region

    neuron_params = {
        "C_m": 250.0, "g_L": 16.67, "E_L": -70.0, "V_th": -50.0,
        "V_reset": -70.0, "t_ref": 2.0, "E_ex": 0.0, "E_in": -80.0,
        "tau_syn_ex": 2.5, "tau_syn_in": 15.0
    }

    E_regions, I_regions = [], []
    for i in range(n_regions):
        E = nest.Create("iaf_cond_exp", N_E_region)
        I = nest.Create("iaf_cond_exp", N_I_region)
        E.set(neuron_params)
        I.set(neuron_params)
        E.V_m = -70.0 + 5.0 * np.random.randn(N_E_region)
        I.V_m = -70.0 + 5.0 * np.random.randn(N_I_region)
        E_regions.append(E)
        I_regions.append(I)

    g_E = 2.0
    g_I = g_ratio * g_E
    delay_local = 1.5
    delay_inter = 3.0

    # External drive
    for region_idx in range(n_regions):
        ext = nest.Create("poisson_generator")
        ext.rate = nu_ext * 1000.0
        nest.Connect(ext, E_regions[region_idx], syn_spec={"weight": g_E, "delay": delay_local})
        nest.Connect(ext, I_regions[region_idx], syn_spec={"weight": g_E, "delay": delay_local})

    # Local recurrent connections
    conn = {"rule": "pairwise_bernoulli", "p": p_conn_local}
    for region_idx in range(n_regions):
        E, I = E_regions[region_idx], I_regions[region_idx]
        nest.Connect(E, E, conn, syn_spec={"weight": g_E, "delay": delay_local})
        nest.Connect(E, I, conn, syn_spec={"weight": g_E, "delay": delay_local})
        nest.Connect(I, E, conn, syn_spec={"weight": g_I, "delay": delay_local})
        nest.Connect(I, I, conn, syn_spec={"weight": g_I, "delay": delay_local})

    # Inter-region connections
    conn_normalized = conn_matrix / np.max(conn_matrix) if np.max(conn_matrix) > 0 else conn_matrix
    for i in range(n_regions):
        for j in range(n_regions):
            if i == j:
                continue
            fc_weight = conn_normalized[i, j]
            if fc_weight > 0.1:
                inter_weight = coupling_strength * g_E * fc_weight
                conn_inter = {"rule": "pairwise_bernoulli", "p": 0.02}
                nest.Connect(E_regions[i], E_regions[j], conn_inter, syn_spec={"weight": inter_weight, "delay": delay_inter})

    # Multimeter recording
    n_rec = max(21, int(record_fraction * N_E_region))
    n_rec = min(n_rec, N_E_region)
    mm = nest.Create("multimeter", params={"interval": 1.0, "record_from": ["V_m"]})
    for region in E_regions:
        nest.Connect(mm, region[:n_rec])

    spike_rec = nest.Create("spike_recorder")
    for E in E_regions:
        nest.Connect(E, spike_rec)

    # Simulate
    nest.Simulate(warmup)
    nest.Simulate(sim_time)

    ev = mm.get("events")
    V_m = np.array(ev["V_m"])
    lfp = V_m.reshape(-1, n_regions*n_rec).mean(axis=1)
    lfp -= lfp.mean()

    fs = 1000.0
    nperseg = min(len(lfp)//2, 16384)
    nperseg = max(nperseg, 8192)
    noverlap = int(15*nperseg//16)
    f, Pxx = welch(lfp, fs=fs, nperseg=nperseg, noverlap=noverlap, window='hann')
    mask = (f>=1) & (f<=40)
    f, Pxx = f[mask], Pxx[mask]

    if smooth_spectrum and len(Pxx) > 10:
        win_len = min(21, len(Pxx)//2*2-1)
        if win_len>=5:
            Pxx = savgol_filter(Pxx, window_length=win_len, polyorder=3)
        Pxx = gaussian_filter1d(Pxx, sigma=0.8)

    Pxx = np.maximum(Pxx, 0)
    Pxx /= Pxx.sum()

    return {"f": f, "Pxx": Pxx, "success": True}

def stitch_spectra(results_by_condition, band_ranges):
    stitched = {}
    band_order = ['delta', 'theta', 'alpha', 'beta', 'gamma']
    for condition, band_results in results_by_condition.items():
        all_f, all_Pxx = [], []
        for band_name in band_order:
            if band_name not in band_results:
                continue
            r = band_results[band_name]
            f, Pxx = r['f'], r['Pxx']
            mask = (f>=band_ranges[band_name][0]) & (f<band_ranges[band_name][1])
            all_f.append(f[mask])
            all_Pxx.append(Pxx[mask])
        if all_f:
            stitched[condition] = {"f": np.concatenate(all_f), "Pxx": np.concatenate(all_Pxx)}
            stitched[condition]["Pxx"] /= stitched[condition]["Pxx"].sum()
    return stitched

# -------------------- Parameters --------------------
DATA_ROOT = "../ds004504/"
BAND_RANGES = {'delta':(1,4), 'theta':(4,8), 'alpha':(8,13), 'beta':(13,30), 'gamma':(30,40)}
N_REGIONS = 19
N_PER_REGION = 52
CONDITIONS = {'HC':6.5, 'AD':2.5}
N_REPEATS = 10

# -------------------- Repeat simulations --------------------
all_results = {cond:{band: [] for band in BAND_RANGES} for cond in CONDITIONS}

for repeat_idx in range(N_REPEATS):
    print(f"\n{'='*50}\nREPEAT {repeat_idx+1}/{N_REPEATS}\n{'='*50}")
    for condition, g_ratio in CONDITIONS.items():
        results_by_band = {}
        for band_name in BAND_RANGES:
            try:
                conn = load_connectivity_matrix(condition, band_name, DATA_ROOT)
                idx, region_conn = select_regions(conn, N_REGIONS)
                res = run_simulation_with_fc(condition, band_name, g_ratio, region_conn,
                                             N_per_region=N_PER_REGION, n_regions=N_REGIONS,
                                             seed=42+repeat_idx*100)
                if res["success"]:
                    results_by_band[band_name] = res
                    all_results[condition][band_name].append(res["Pxx"])
            except Exception as e:
                print(f"Error ({condition}-{band_name}): {e}")

# -------------------- Compute mean ± std --------------------
stitched_mean_std = {}
for condition in CONDITIONS:
    # stack across repeats
    stacked = []
    for band_name in BAND_RANGES:
        band_list = all_results[condition][band_name]
        # Only include if all repeats succeeded
        if len(band_list) == N_REPEATS:
            stacked.append(np.vstack(band_list))
    if stacked:
        Pxx_all = np.vstack([band for sublist in stacked for band in sublist])
        # Assume same f-axis from last repeat
        f_axis = res["f"]
        stitched_mean_std[condition] = {"f": f_axis,
                                        "mean": np.mean(Pxx_all, axis=0),
                                        "std": np.std(Pxx_all, axis=0)}

# -------------------- Plot --------------------
fig, ax = plt.subplots(figsize=(12,6))
colors = {"AD":"#90EE90", "HC":"#A9A9A9"}

for condition, data in stitched_mean_std.items():
    ax.plot(data["f"], data["mean"], color=colors[condition], linewidth=2.5, label=condition)
    ax.fill_between(data["f"], data["mean"]-data["std"], data["mean"]+data["std"], color=colors[condition], alpha=0.25)

# Band boundaries
for b in [4,8,13,30]:
    ax.axvline(b, color="gray", linestyle="--", linewidth=1.0, alpha=0.4)

# Band labels
y_max = ax.get_ylim()[1]
band_centers = [(1+4)/2, (4+8)/2, (8+13)/2, (13+30)/2, (30+40)/2]
band_names = ['Delta', 'Theta', 'Alpha', 'Beta', 'Gamma']
for c, n in zip(band_centers, band_names):
    ax.text(c, y_max*0.95, n, ha='center', fontsize=10, style='italic', color='gray', alpha=0.7)

ax.set_xlabel("Frequency (Hz)", fontsize=14, fontweight="bold")
ax.set_ylabel("Relative Power", fontsize=14, fontweight="bold")
ax.set_xlim([1,40])
ax.grid(alpha=0.3)
ax.legend(fontsize=11, loc='upper right')
plt.tight_layout()
plt.savefig("improved_ei_fc_spectrum_mean_std.png", dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved: improved_ei_fc_spectrum_mean_std.png")

