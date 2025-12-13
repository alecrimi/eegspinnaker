# ============================================================
# PyNEST – E/I imbalance model (Martínez-Cañada et al. style)
# EEG/MEG proxy from synaptic currents
# ============================================================

import nest
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch

def run_simulation(condition, g_ratio,
                   N_total=400,
                   frac_exc=0.8,
                   p_conn=0.1,
                   nu_ext=8.0,
                   sim_time=6000.0,
                   warmup=2000.0,
                   seed=42):
    """
    Run a simple E/I network with synaptic current LFP proxy.
    """

    # --------------------
    # NEST kernel setup
    # --------------------
    nest.ResetKernel()
    nest.resolution = 0.1
    nest.set_verbosity("M_WARNING")
    
    # Use the provided seed (different for each condition)
    np.random.seed(seed)
    nest.SetKernelStatus({"rng_seed": seed})

    # --------------------
    # Populations
    # --------------------
    N_E = int(frac_exc * N_total)
    N_I = N_total - N_E

    E = nest.Create("iaf_psc_exp", N_E)
    I = nest.Create("iaf_psc_exp", N_I)

    for pop in (E, I):
        pop.V_m = -70 + 5 * np.random.randn(len(pop))
        pop.V_th = -50 + 2 * np.random.randn(len(pop))
        pop.tau_syn_ex = 2.0
        pop.tau_syn_in = 8.0
        pop.t_ref = 2.0

    # --------------------
    # Synaptic strengths
    # --------------------
    g_E = 3.0
    g_I = g_ratio * g_E
    delay = 1.5

    print(f"\n[{condition}] g_I/g_E = {g_ratio:.2f}, seed = {seed}")

    # --------------------
    # External drive
    # --------------------
    ext = nest.Create("poisson_generator")
    ext.rate = nu_ext * 1000.0  # Hz

    nest.Connect(ext, E, syn_spec={"weight": g_E, "delay": delay})
    nest.Connect(ext, I, syn_spec={"weight": g_E, "delay": delay})

    # --------------------
    # Recurrent connections
    # --------------------
    conn = {"rule": "pairwise_bernoulli", "p": p_conn}

    nest.Connect(E, E, conn, syn_spec={"weight": g_E, "delay": delay})
    nest.Connect(E, I, conn, syn_spec={"weight": g_E, "delay": delay})
    nest.Connect(I, E, conn, syn_spec={"weight": -g_I, "delay": delay})
    nest.Connect(I, I, conn, syn_spec={"weight": -g_I, "delay": delay})

    # --------------------
    # Multimeter for synaptic currents
    # --------------------
    mm = nest.Create("multimeter")
    mm.set({
        "interval": 1.0,
        "record_from": ["I_syn_ex", "I_syn_in"]
    })

    n_rec = min(40, N_E)
    nest.Connect(mm, E[:n_rec])

    # --------------------
    # Spike recorder (optional)
    # --------------------
    spike_rec = nest.Create("spike_recorder")
    nest.Connect(E, spike_rec)

    # --------------------
    # Simulate
    # --------------------
    nest.Simulate(warmup)
    nest.Simulate(sim_time)

    # --------------------
    # EEG / LFP proxy
    # --------------------
    ev = mm.get("events")
    times = np.array(ev["times"])
    senders = np.array(ev["senders"])
    I_ex = np.array(ev["I_syn_ex"])
    I_in = np.array(ev["I_syn_in"])

    # Filter out warmup period
    mask = times > warmup
    times_filtered = times[mask]
    senders_filtered = senders[mask]
    I_ex_filtered = I_ex[mask]
    I_in_filtered = I_in[mask]

    # Get unique time points and neurons
    unique_times = np.sort(np.unique(times_filtered))
    unique_neurons = np.sort(np.unique(senders_filtered))
    n_times = len(unique_times)
    n_neurons = len(unique_neurons)
    
    print(f"    Recording from {n_neurons} neurons over {n_times} time points")

    # NEST records data chronologically: neuron1@t1, neuron2@t1, ..., neuron1@t2, ...
    # So data should be in order if recording is complete
    expected_length = n_times * n_neurons
    
    if len(times_filtered) != expected_length:
        print(f"    WARNING: Expected {expected_length} points, got {len(times_filtered)}")
        # Handle missing data by explicit indexing
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
        # Reshape directly - NEST records in blocks per time point
        I_ex_matrix = I_ex_filtered.reshape(n_times, n_neurons)
        I_in_matrix = I_in_filtered.reshape(n_times, n_neurons)

    # LFP proxy: average across neurons at each time point
    lfp = I_ex_matrix.mean(axis=1) - I_in_matrix.mean(axis=1)
    lfp -= lfp.mean()

    print(f"    LFP: {len(lfp)} samples, range [{lfp.min():.2f}, {lfp.max():.2f}] pA")

    # --------------------
    # Power spectrum
    # --------------------
    fs = 1000.0
    nperseg = min(4096, len(lfp) // 4)

    f, Pxx = welch(lfp, fs=fs,
                   nperseg=nperseg,
                   noverlap=3 * nperseg // 4)

    band = (f >= 1) & (f <= 40)
    f = f[band]
    Pxx = Pxx[band]
    Pxx /= Pxx.sum()  # relative power

    return {
        "condition": condition,
        "g_ratio": g_ratio,
        "f": f,
        "Pxx": Pxx,
        "lfp": lfp  # Keep LFP for verification
    }


# ============================================================
# Main execution
# ============================================================

conditions = [
    ("AD", 2.5, 42),    # Low inhibition - different seed
    ("MCI", 3.5, 123),  # Medium inhibition - different seed  
    ("HC", 6.5, 456),   # High inhibition - different seed
]

print("=" * 60)
print("Running E/I Balance Simulations")
print("=" * 60)

results = []
for name, g, seed in conditions:
    res = run_simulation(name, g, seed=seed)
    if res is not None:
        results.append(res)

# --------------------
# Verification: Check that LFPs are different
# --------------------
print("\n" + "=" * 60)
print("LFP Verification (checking first 100 samples):")
print("=" * 60)
for i, r in enumerate(results):
    print(f"{r['condition']:3s}: {r['lfp'][:5]}")

# Check if any two are identical
if len(results) >= 2:
    for i in range(len(results)):
        for j in range(i+1, len(results)):
            if np.allclose(results[i]['lfp'], results[j]['lfp']):
                print(f"\n⚠️  WARNING: {results[i]['condition']} and {results[j]['condition']} have identical LFPs!")
            else:
                corr = np.corrcoef(results[i]['lfp'], results[j]['lfp'])[0, 1]
                print(f"✓ {results[i]['condition']} vs {results[j]['condition']}: correlation = {corr:.3f}")

# --------------------
# Plot
# --------------------
if results:
    fig, axes = plt.subplots(2, 1, figsize=(10, 10))
    
    # Plot 1: Power spectra
    ax = axes[0]
    colors = {'AD': 'red', 'MCI': 'orange', 'HC': 'blue'}
    for r in results:
        ax.plot(r["f"], r["Pxx"], label=r["condition"], 
                linewidth=2.5, color=colors.get(r["condition"], 'gray'))
    ax.set_xlabel("Frequency (Hz)", fontsize=12)
    ax.set_ylabel("Relative power", fontsize=12)
    ax.set_xlim(1, 40)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=11)
    ax.set_title("E/I Balance Effects on EEG Power Spectrum", fontsize=13)
    
    # Plot 2: Example LFP traces (first 2 seconds)
    ax = axes[1]
    fs = 1000.0
    for r in results:
        t = np.arange(len(r['lfp'][:2000])) / fs
        ax.plot(t, r['lfp'][:2000], label=r["condition"], 
                linewidth=1, alpha=0.7, color=colors.get(r["condition"], 'gray'))
    ax.set_xlabel("Time (s)", fontsize=12)
    ax.set_ylabel("LFP proxy (pA)", fontsize=12)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=11)
    ax.set_title("Example LFP Traces (first 2 seconds)", fontsize=13)
    
    plt.tight_layout()
    plt.savefig("EI_EEG_proxy_corrected.png", dpi=300)
    print("\n✓ Plot saved as EI_EEG_proxy_corrected.png")
    plt.show()
else:
    print("\n✗ No successful simulations to plot!")
