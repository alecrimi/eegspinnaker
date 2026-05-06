# ============================================================
# PyNEST – Improved E/I imbalance with FC-based connectivity
# 10 repetitions with mean/std and overlay plots
# ============================================================

import nest
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch, savgol_filter
from scipy.ndimage import gaussian_filter1d
import os


def load_connectivity_matrix(group, band, data_root="./"):
    """Load functional connectivity matrix from PLV data."""
    conn_file = os.path.join(data_root, group, f'connectivity_{band}', f'average_{band}_plv.npy')
    
    if not os.path.exists(conn_file):
        raise FileNotFoundError(f"Connectivity file not found: {conn_file}")
    
    conn_matrix = np.load(conn_file)
    np.fill_diagonal(conn_matrix, 0)
    
    print(f"    Loaded {band} FC: range [{conn_matrix.min():.3f}, {conn_matrix.max():.3f}]")
    
    return conn_matrix


def select_regions(conn_matrix, n_regions=19):
    """Select regions with highest connectivity."""
    node_strength = np.sum(np.abs(conn_matrix), axis=1)
    selected_indices = np.argsort(node_strength)[-n_regions:]
    region_conn = conn_matrix[np.ix_(selected_indices, selected_indices)]
    
    return selected_indices, region_conn


def run_simulation_with_fc(condition, 
                           band_name,
                           g_ratio,
                           conn_matrix,
                           N_per_region=100,
                           n_regions=19,
                           frac_exc=0.8,
                           p_conn_local=0.2,
                           coupling_strength=1.5,
                           nu_ext=15.0,
                           sim_time=10000.0,
                           warmup=3000.0,
                           seed=42,
                           record_fraction=0.2,
                           smooth_spectrum=True):
    """
    Run E/I network with enhanced parameters for reliable 10 Hz oscillations.
    """

    nest.ResetKernel()
    nest.resolution = 0.1
    nest.set_verbosity("M_WARNING")
    
    rng_seed = seed + hash(condition + band_name) % 1000
    np.random.seed(rng_seed)
    nest.SetKernelStatus({"rng_seed": rng_seed})

    print(f"\n[{condition} - {band_name}] g_I/g_E = {g_ratio:.2f}, seed = {rng_seed}")

    N_E_region = int(N_per_region * frac_exc)
    N_I_region = N_per_region - N_E_region
    
    neuron_params = {
        "C_m": 250.0,          
        "g_L": 16.67,          
        "E_L": -70.0,          
        "V_th": -50.0,         
        "V_reset": -70.0,      
        "t_ref": 2.0,          
        "E_ex": 0.0,           
        "E_in": -80.0,         
        "tau_syn_ex": 2.5,     
        "tau_syn_in": 15.0,    
    }
    
    E_regions = []
    I_regions = []
    
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

    for region_idx in range(n_regions):
        ext = nest.Create("poisson_generator")
        ext.rate = nu_ext * 1000.0
        
        nest.Connect(ext, E_regions[region_idx], syn_spec={"weight": g_E, "delay": delay_local})
        nest.Connect(ext, I_regions[region_idx], syn_spec={"weight": g_E, "delay": delay_local})

    conn = {"rule": "pairwise_bernoulli", "p": p_conn_local}
    
    for region_idx in range(n_regions):
        E = E_regions[region_idx]
        I = I_regions[region_idx]
        
        nest.Connect(E, E, conn, syn_spec={"weight": g_E, "delay": delay_local})
        nest.Connect(E, I, conn, syn_spec={"weight": g_E, "delay": delay_local})
        nest.Connect(I, E, conn, syn_spec={"weight": g_I, "delay": delay_local})
        nest.Connect(I, I, conn, syn_spec={"weight": g_I, "delay": delay_local})

    conn_normalized = conn_matrix / np.max(conn_matrix) if np.max(conn_matrix) > 0 else conn_matrix
    
    for i in range(n_regions):
        for j in range(n_regions):
            if i == j:
                continue
            
            fc_weight = conn_normalized[i, j]
            
            if fc_weight > 0.1:
                inter_weight = coupling_strength * g_E * fc_weight
                conn_inter = {"rule": "pairwise_bernoulli", "p": 0.02}
                
                nest.Connect(E_regions[i], E_regions[j], 
                           conn_inter,
                           syn_spec={"weight": inter_weight, "delay": delay_inter})

    n_rec_per_region = max(21, int(record_fraction * N_E_region))
    n_rec_per_region = min(n_rec_per_region, N_E_region)
    
    mm = nest.Create("multimeter")
    mm.set({
        "interval": 1.0,
        "record_from": ["g_ex", "g_in", "V_m"]
    })
    
    for r in range(n_regions):
        nest.Connect(mm, E_regions[r][:n_rec_per_region])
    print(f"    Recording from {n_rec_per_region}/{N_E_region} neurons per region ({100*n_rec_per_region/N_E_region:.1f}%)")

    spike_rec = nest.Create("spike_recorder")
    for E in E_regions:
        nest.Connect(E, spike_rec)

    nest.Simulate(warmup)
    nest.Simulate(sim_time)

    ev = mm.get("events")
    times = np.array(ev["times"])
    senders = np.array(ev["senders"])
    g_ex = np.array(ev["g_ex"])
    g_in = np.array(ev["g_in"])
    V_m = np.array(ev["V_m"])
    
    E_ex = 0.0
    E_in = -80.0
    
    I_ex = g_ex * (V_m - E_ex)
    I_in = g_in * (V_m - E_in)

    mask = times > warmup
    times_filtered = times[mask]
    senders_filtered = senders[mask]
    I_ex_filtered = I_ex[mask]
    I_in_filtered = I_in[mask]

    if len(times_filtered) < 1000:
        print(f"    WARNING: Insufficient data ({len(times_filtered)} points)")
        return {'success': False}

    unique_times = np.sort(np.unique(times_filtered))
    unique_neurons = np.sort(np.unique(senders_filtered))
    n_times = len(unique_times)
    n_neurons = n_regions * n_rec_per_region
    
    print(f"    Recording from {n_neurons} neurons over {n_times} time points")

    expected_length = n_times * n_neurons
    
    if len(times_filtered) != expected_length:
        print(f"    WARNING: Expected {expected_length} points, got {len(times_filtered)}")
        I_ex_matrix = np.zeros((n_times, n_neurons))
        I_in_matrix = np.zeros((n_times, n_neurons))
        
        neuron_to_idx = {nid: idx for idx, nid in enumerate(unique_neurons)}
        time_to_idx = {t: idx for idx, t in enumerate(unique_times)}
        
        for i in range(len(times_filtered)):
            t_idx = time_to_idx[times_filtered[i]]
            n_idx = neuron_to_idx[senders_filtered[i]]
            I_ex_matrix[t_idx, n_idx] = I_ex_filtered[i]
            I_in_matrix[t_idx, n_idx] = I_in_filtered[i]
    else:
        I_ex_matrix = I_ex_filtered.reshape(n_times, n_neurons)
        I_in_matrix = I_in_filtered.reshape(n_times, n_neurons)

    #lfp = I_ex_matrix.mean(axis=1) - I_in_matrix.mean(axis=1)
    lfp = I_ex_matrix.mean(axis=1) + I_in_matrix.mean(axis=1)
    lfp -= lfp.mean()

    print(f"    LFP: {len(lfp)} samples, range [{lfp.min():.2f}, {lfp.max():.2f}] pA")

    fs = 1000.0
    
    nperseg = min(len(lfp) // 2, 16384)
    nperseg = max(nperseg, 8192)
    noverlap = int(15 * nperseg // 16)

    f, Pxx = welch(lfp, fs=fs,
                   nperseg=nperseg,
                   noverlap=noverlap,
                   window='hann',
                   detrend='constant')

    band = (f >= 1) & (f <= 40)
    f = f[band]
    Pxx = Pxx[band]
    
    if smooth_spectrum and len(Pxx) > 10:
        window_length = min(21, len(Pxx) // 2 * 2 - 1)
        if window_length >= 10:
            Pxx = savgol_filter(Pxx, window_length=window_length, polyorder=3)
        
        sigma = 0.8
        Pxx = gaussian_filter1d(Pxx, sigma=sigma)
        
        print(f"    Applied dual smoothing (SG window={window_length}, Gaussian σ={sigma})")
    
    Pxx = np.maximum(Pxx, 0)
    Pxx /= Pxx.sum()

    spikes = spike_rec.get("events")
    spike_times = spikes["times"]
    spike_times = spike_times[spike_times > warmup]
    total_neurons = n_regions * N_E_region
    spike_rate = len(spike_times) / (total_neurons * sim_time / 1000.0)
    
    print(f"    Spike rate: {spike_rate:.2f} Hz, Welch: nperseg={nperseg}")

    return {
        "condition": condition,
        "band": band_name,
        "g_ratio": g_ratio,
        "f": f,
        "Pxx": Pxx,
        "lfp": lfp,
        "spike_rate": spike_rate,
        "success": True
    }

def stitch_spectra_with_blending(results_by_condition, band_ranges, blend_width=1.0):
    """Stitch spectra without tapering to avoid artificial zeros at band boundaries."""
    stitched = {}
    
    for condition, band_results in results_by_condition.items():
        all_f = []
        all_Pxx = []
        
        band_order = ['delta', 'theta', 'alpha', 'beta', 'gamma']
        
        for band_name in band_order:
            if band_name not in band_results:
                print(f"    Warning: Missing {band_name} for {condition}")
                continue
            
            result = band_results[band_name]
            f = result['f']
            Pxx = result['Pxx']
            
            low, high = band_ranges[band_name]
            mask = (f >= low) & (f < high)
            
            all_f.append(f[mask])
            all_Pxx.append(Pxx[mask])
        
        if all_f:
            stitched_f = np.concatenate(all_f)
            stitched_Pxx = np.concatenate(all_Pxx)
            
            # Light Gaussian smoothing only — no tapering, no artificial zeros
            stitched_Pxx = gaussian_filter1d(stitched_Pxx, sigma=2.0)
            
            # Renormalize
            stitched_Pxx = np.maximum(stitched_Pxx, 0)
            stitched_Pxx /= stitched_Pxx.sum()
            
            stitched[condition] = {
                'f': stitched_f,
                'Pxx': stitched_Pxx
            }
    
    return stitched
    

def interpolate_to_common_grid(f, Pxx, f_common):
    """Interpolate spectrum to a common frequency grid."""
    return np.interp(f_common, f, Pxx)


def plot_mean_std(all_runs_stitched, conditions, colors, band_boundaries, band_centers, band_names):
    """Plot mean ± std across 10 runs for each condition."""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Build common frequency grid
    f_min, f_max = 1.0, 40.0
    f_common = np.linspace(f_min, f_max, 500)
    
    for condition in conditions:
        runs_Pxx = []
        for run_data in all_runs_stitched:
            if condition in run_data:
                Pxx_interp = interpolate_to_common_grid(
                    run_data[condition]['f'],
                    run_data[condition]['Pxx'],
                    f_common
                )
                runs_Pxx.append(Pxx_interp)
        
        if len(runs_Pxx) == 0:
            continue
        
        runs_array = np.array(runs_Pxx)
        mean_Pxx = runs_array.mean(axis=0)
        std_Pxx = runs_array.std(axis=0)
        
        color = colors[condition]
        ax.plot(f_common, mean_Pxx, label=condition, linewidth=2.5, color=color, alpha=0.9)
        ax.fill_between(f_common,
                mean_Pxx - std_Pxx,
                mean_Pxx + std_Pxx,
                color=color, alpha=0.25)
    
    for boundary in band_boundaries:
        ax.axvline(x=boundary, color='gray', linestyle='--', linewidth=1.0, alpha=0.4)
    
    y_max = ax.get_ylim()[1]
    for center, name in zip(band_centers, band_names):
        ax.text(center, y_max * 0.95, name,
                horizontalalignment='center', fontsize=10,
                style='italic', color='gray', alpha=0.7)
    
    ax.set_xlabel("Frequency (Hz)", fontsize=14, fontweight='bold')
    ax.set_ylabel("Relative Power", fontsize=14, fontweight='bold')
    ax.set_xlim([1, 40])
    #ax.set_title("E/I Balance Spectrum – Mean ± 1 SD (n=10 runs)", fontsize=14, fontweight='bold')
    ax.grid(alpha=0.3, linewidth=0.5)
    ax.legend(fontsize=11, loc='upper right')
    
    plt.tight_layout()
    plt.savefig("ei_fc_spectrum_mean_std.png", dpi=300, bbox_inches='tight')
    print("✓ Saved: ei_fc_spectrum_mean_std.png")
    plt.close()


def plot_overlay(all_runs_stitched, conditions, colors, band_boundaries, band_centers, band_names):
    """Plot overlay of all 10 runs per condition."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)
    
    f_min, f_max = 1.0, 40.0
    f_common = np.linspace(f_min, f_max, 500)
    
    for ax_idx, condition in enumerate(conditions):
        ax = axes[ax_idx]
        color = colors[condition]
        
        runs_Pxx = []
        for run_idx, run_data in enumerate(all_runs_stitched):
            if condition in run_data:
                Pxx_interp = interpolate_to_common_grid(
                    run_data[condition]['f'],
                    run_data[condition]['Pxx'],
                    f_common
                )
                runs_Pxx.append(Pxx_interp)
                ax.plot(f_common, Pxx_interp,
                        linewidth=1.2, color=color, alpha=0.35,
                        label=f"Run {run_idx+1}" if run_idx == 0 else "_nolegend_")
        
        if runs_Pxx:
            runs_array = np.array(runs_Pxx)
            mean_Pxx = runs_array.mean(axis=0)
            ax.plot(f_common, mean_Pxx, linewidth=3.0, color=color,
                    alpha=1.0, label="Mean", zorder=5)
        
        for boundary in band_boundaries:
            ax.axvline(x=boundary, color='gray', linestyle='--', linewidth=1.0, alpha=0.4)
        
        y_max = ax.get_ylim()[1]
        for center, name in zip(band_centers, band_names):
            ax.text(center, y_max * 0.95, name,
                    horizontalalignment='center', fontsize=10,
                    style='italic', color='gray', alpha=0.7)
        
        ax.set_xlabel("Frequency (Hz)", fontsize=13, fontweight='bold')
        if ax_idx == 0:
            ax.set_ylabel("Relative Power", fontsize=13, fontweight='bold')
        ax.set_xlim([1, 40])
        ax.set_title(f"{condition} – All 10 Runs", fontsize=13, fontweight='bold', color=color)
        ax.grid(alpha=0.3, linewidth=0.5)
        
        from matplotlib.lines import Line2D
        legend_elements = [    Line2D([0], [0], color=colors[c], linewidth=2.5, label=c)    for c in conditions]
        ax.legend(handles=legend_elements, fontsize=11, loc='upper right')
    
    fig.suptitle("E/I Balance Spectrum – Overlay of 10 Runs", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig("ei_fc_spectrum_overlay.png", dpi=300, bbox_inches='tight')
    print("✓ Saved: ei_fc_spectrum_overlay.png")
    plt.close()


# ============================================================
# Main execution
# ============================================================

DATA_ROOT = "../ds004504/"
BAND_RANGES = {
    'delta': (1, 4),
    'theta': (4, 8),
    'alpha': (8, 13),
    'beta': (13, 30),
    'gamma': (30, 40)
}
N_REGIONS = 19
N_PER_REGION = 120
N_RUNS = 1

CONDITIONS = {
    'HC': 6.5,    
    'AD': 2.5
}

COLORS = {
    "AD": "#90EE90",
    "HC": "#A9A9A9",
}

BAND_BOUNDARIES = [4, 8, 13, 30]
BAND_CENTERS = [(1+4)/2, (4+8)/2, (8+13)/2, (13+30)/2, (30+40)/2]
BAND_NAMES = ['Delta', 'Theta', 'Alpha', 'Beta', 'Gamma']

print("="*70)
print("IMPROVED E/I BALANCE – 10 REPETITIONS")
print("="*70)
print(f"  • {N_RUNS} independent runs per condition")
print(f"  • Plot 1: Mean ± 1 SD")
print(f"  • Plot 2: Overlay of all runs")
print("="*70)

# Store stitched results for all runs
all_runs_stitched = []  # list of dicts: {condition: {f, Pxx}}

for run_idx in range(N_RUNS):
    run_seed = 42 + run_idx * 100  # Different seed per run

    print(f"\n{'='*70}")
    print(f"RUN {run_idx + 1} / {N_RUNS}  (seed offset = {run_seed})")
    print(f"{'='*70}")
    
    results_by_condition = {condition: {} for condition in CONDITIONS.keys()}
    
    for condition, g_ratio in CONDITIONS.items():
        print(f"\n  CONDITION: {condition}")
        
        for band_name, (low, high) in BAND_RANGES.items():
            print(f"    [{band_name.upper()}] {low}-{high} Hz")
            
            try:
                conn_matrix = load_connectivity_matrix(condition, band_name, DATA_ROOT)
                selected_idx, region_conn = select_regions(conn_matrix, N_REGIONS)
                
                result = run_simulation_with_fc(
                    condition=condition,
                    band_name=band_name,
                    g_ratio=g_ratio,
                    conn_matrix=region_conn,
                    N_per_region=N_PER_REGION,
                    n_regions=N_REGIONS,
                    coupling_strength=1.5,
                    sim_time=10000.0,
                    warmup=3000.0,
                    seed=run_seed,          # <-- Varies per run
                    record_fraction=0.2,
                    smooth_spectrum=True
                )
                
                if result.get('success', False):
                    results_by_condition[condition][band_name] = result
                    print(f"      ✓ spike rate: {result['spike_rate']:.2f} Hz")
                else:
                    print(f"      ✗ Failed")
            
            except Exception as e:
                print(f"      ✗ Error: {e}")
    
    # Stitch this run
    run_stitched = stitch_spectra_with_blending(results_by_condition, BAND_RANGES, blend_width=1.0)
    all_runs_stitched.append(run_stitched)
    
    for condition, data in run_stitched.items():
        alpha_mask = (data['f'] >= 8) & (data['f'] <= 13)
        if np.any(alpha_mask):
            peak_freq = data['f'][alpha_mask][np.argmax(data['Pxx'][alpha_mask])]
            peak_power = np.max(data['Pxx'][alpha_mask])
            print(f"    {condition} alpha peak: {peak_freq:.1f} Hz (power: {peak_power:.4f})")

# ============================================================
# Plot 1: Mean ± SD
# ============================================================
print(f"\n{'='*70}")
print("PLOTTING: Mean ± SD")
print(f"{'='*70}")
plot_mean_std(all_runs_stitched, list(CONDITIONS.keys()), COLORS,
              BAND_BOUNDARIES, BAND_CENTERS, BAND_NAMES)

# ============================================================
# Plot 2: Overlay
# ============================================================
print(f"\n{'='*70}")
print("PLOTTING: Overlay of all runs")
print(f"{'='*70}")
plot_overlay(all_runs_stitched, list(CONDITIONS.keys()), COLORS,
             BAND_BOUNDARIES, BAND_CENTERS, BAND_NAMES)

print("\n" + "="*70)
print("COMPLETE – Outputs:")
print("  • ei_fc_spectrum_mean_std.png")
print("  • ei_fc_spectrum_overlay.png")
print("="*70)
