# ============================================================
# PyNEST – E/I imbalance model (Conductance-based LIF)
# EEG/MEG proxy from synaptic currents
# ============================================================
import nest
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch

def run_simulation(condition, g_ratio, N_total=5000, frac_exc=0.8, p_conn=0.2,
                   nu_ext=3.0, sim_time=6000.0, warmup=2000.0, seed=42):
    """
    Run a simple E/I network with conductance-based neurons and synaptic current LFP proxy.
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
    
    # Conductance-based LIF neurons
    E = nest.Create("iaf_cond_exp", N_E)
    I = nest.Create("iaf_cond_exp", N_I)
    
    # Conductance-based neuron parameters
    # Standard parameters for cortical neurons
    neuron_params = {
        "C_m": 250.0,          # pF - membrane capacitance
        "g_L": 16.67,          # nS - leak conductance
        "E_L": -70.0,          # mV - leak reversal potential
        "V_th": -50.0,         # mV - spike threshold
        "V_reset": -70.0,      # mV - reset potential
        "t_ref": 2.0,          # ms - refractory period
        "E_ex": 0.0,           # mV - excitatory reversal potential
        "E_in": -80.0,         # mV - inhibitory reversal potential
        "tau_syn_ex": 2.0,     # ms - excitatory synaptic time constant
        "tau_syn_in": 8.0,     # ms - inhibitory synaptic time constant
    }
    
    # Set parameters for both populations
    for pop in (E, I):
        pop.set(neuron_params)
        # Initialize membrane potentials with variability
        pop.V_m = -70.0 + 5.0 * np.random.randn(len(pop))
    
    # --------------------
    # Synaptic strengths (conductances in nS)
    # --------------------
    g_E = 2.0  # nS - excitatory synaptic conductance
    g_I = g_ratio * g_E  # nS - inhibitory synaptic conductance
    delay = 1.5
    
    print(f"\n[{condition}] g_I/g_E = {g_ratio:.2f}, seed = {seed}")
    print(f"  g_E = {g_E:.2f} nS, g_I = {g_I:.2f} nS")
    
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
    nest.Connect(I, E, conn, syn_spec={"weight": g_I, "delay": delay})  # Note: positive weight
    nest.Connect(I, I, conn, syn_spec={"weight": g_I, "delay": delay})  # Conductance is always positive
    
    # --------------------
    # Multimeter for synaptic conductances and membrane potential
    # --------------------
    mm = nest.Create("multimeter")
    mm.set({
        "interval": 1.0,
        # For conductance-based neurons, record conductances and V_m
        "record_from": ["g_ex", "g_in", "V_m"]
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
    g_ex = np.array(ev["g_ex"])  # nS - excitatory conductance
    g_in = np.array(ev["g_in"])  # nS - inhibitory conductance
    V_m = np.array(ev["V_m"])    # mV - membrane potential
    
    # Calculate synaptic currents from conductances and driving forces
    # I_syn = g_syn * (V_m - E_reversal)
    E_ex = 0.0   # mV - excitatory reversal potential
    E_in = -80.0 # mV - inhibitory reversal potential
    
    I_ex = g_ex * (V_m - E_ex)  # pA (since g in nS, V in mV)
    I_in = g_in * (V_m - E_in)  # pA
    
    # Filter out warmup period
    mask = times > warmup
    times_filtered = times[mask]
    senders_filtered = senders[mask]
    I_ex_filtered = I_ex[mask]
    I_in_filtered = I_in[mask]
    V_m_filtered = V_m[mask]
    
    # Get unique time points and neurons
    unique_times = np.sort(np.unique(times_filtered))
    unique_neurons = np.sort(np.unique(senders_filtered))
    n_times = len(unique_times)
    n_neurons = len(unique_neurons)
    
    print(f"  Recording from {n_neurons} neurons over {n_times} time points")
    
    # NEST records data chronologically: neuron1@t1, neuron2@t1, ..., neuron1@t2, ...
    expected_length = n_times * n_neurons
    if len(times_filtered) != expected_length:
        print(f"  WARNING: Expected {expected_length} points, got {len(times_filtered)}")
        # Handle missing data by explicit indexing
        I_ex_matrix = np.zeros((n_times, n_neurons))
        I_in_matrix = np.zeros((n_times, n_neurons))
        V_m_matrix = np.zeros((n_times, n_neurons))
        
        neuron_to_idx = {nid: idx for idx, nid in enumerate(unique_neurons)}
        time_to_idx = {t: idx for idx, t in enumerate(unique_times)}
        
        for i in range(len(times_filtered)):
            t_idx = time_to_idx[times_filtered[i]]
            n_idx = neuron_to_idx[senders_filtered[i]]
            I_ex_matrix[t_idx, n_idx] = I_ex_filtered[i]
            I_in_matrix[t_idx, n_idx] = I_in_filtered[i]
            V_m_matrix[t_idx, n_idx] = V_m_filtered[i]
    else:
        # Reshape directly - NEST records in blocks per time point
        I_ex_matrix = I_ex_filtered.reshape(n_times, n_neurons)
        I_in_matrix = I_in_filtered.reshape(n_times, n_neurons)
        V_m_matrix = V_m_filtered.reshape(n_times, n_neurons)
    
    # LFP proxy: average synaptic currents across neurons at each time point
    # I_syn_ex is inward (negative when V_m < 0), I_syn_in is outward (positive when V_m > -80)
    # LFP convention: I_ex - I_in (excitatory minus inhibitory currents)
    lfp = I_ex_matrix.mean(axis=1) - I_in_matrix.mean(axis=1)
    lfp -= lfp.mean()
    
    print(f"  LFP: {len(lfp)} samples, range [{lfp.min():.2f}, {lfp.max():.2f}] pA")
    print(f"  Mean V_m: {V_m_matrix.mean():.2f} mV, std: {V_m_matrix.std():.2f} mV")
    
    # Get spike statistics
    spike_events = spike_rec.get("events")
    spike_times = spike_events["times"]
    spike_times = spike_times[spike_times > warmup]
    firing_rate = len(spike_times) / (sim_time / 1000.0) / N_E
    print(f"  Mean firing rate (E): {firing_rate:.2f} Hz")
    
    # --------------------
    # Power spectrum
    # --------------------
    fs = 1000.0
    nperseg = min(4096, len(lfp) // 4)
    f, Pxx = welch(lfp, fs=fs, nperseg=nperseg, noverlap=3 * nperseg // 4)
    
    band = (f >= 1) & (f <= 40)
    f = f[band]
    Pxx = Pxx[band]
    Pxx /= Pxx.sum()  # relative power
    
    return {
        "condition": condition,
        "g_ratio": g_ratio,
        "f": f,
        "Pxx": Pxx,
        "lfp": lfp,
        "firing_rate": firing_rate
    }


# ============================================================
# Main execution
# ============================================================
conditions = [
    ("AD", 2.5, 42),     # Low inhibition - different seed
    ("MCI", 3.5, 123),   # Medium inhibition - different seed
    ("HC", 6.5, 456),    # High inhibition - different seed
]

print("=" * 60)
print("Running E/I Balance Simulations (Conductance-based)")
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
print("LFP Verification (checking first 5 samples):")
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
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    # Power spectra with new colors
    colors = {
        "AD": "#90EE90",
        "MCI": "#FFD700",
        "HC": "#A9A9A9",
    }
    
    for r in results:
        ax.plot(r["f"], r["Pxx"], 
                #label=f"{r['condition']} ({r['firing_rate']:.1f} Hz)", 
                label=f"{r['condition']}", 
                linewidth=2.5, 
                color=colors.get(r["condition"], 'gray'))
    
    # Add vertical dashed lines for frequency band boundaries
    # Delta: 0-4 Hz, Theta: 4-8 Hz, Alpha: 8-13 Hz, Beta: 13-30 Hz, Gamma: 30-45 Hz
    band_boundaries = [4, 8, 13, 30]
    for boundary in band_boundaries:
        ax.axvline(x=boundary, color='gray', linestyle='--', linewidth=1.5, alpha=0.6)
    
    # Add band labels at the top
    y_max = ax.get_ylim()[1]
    band_centers = [(1+4)/2, (4+8)/2, (8+13)/2, (13+30)/2, (30+40)/2]
    band_names = ['Delta', 'Theta', 'Alpha', 'Beta', 'Gamma']
    
    for center, name in zip(band_centers, band_names):
        if center <= 40:  # Only show labels within x-axis range
            ax.text(center, y_max * 0.95, name,
                   horizontalalignment='center',
                   fontsize=10, style='italic',
                   color='gray', alpha=0.7)
    
    ax.set_xlabel("Frequency (Hz)", fontsize=12)
    ax.set_ylabel("Relative power", fontsize=12)
    ax.set_xlim(1, 40)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=11)
    ax.set_title("E/I Balance Effects on EEG Power Spectrum (Conductance-based LIF)", fontsize=13)
    
    plt.tight_layout()
    plt.savefig("EI_EEG_proxy_conductance.png", dpi=300)
    print("\n✓ Plot saved as EI_EEG_proxy_conductance.png")
    plt.show()
else:
    print("\n✗ No successful simulations to plot!")
