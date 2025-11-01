# ============================================================
# 
# PyNEST prototype - Martinez-Cañada et al. 2023 replication
# E/I imbalance in Alzheimer's disease
# ============================================================

import nest
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.signal import welch

def run_simulation(condition_name, g_ratio, N_total=200, frac_exc=0.8, p_conn=0.15, 
                   nu_ext=5.0, sim_time=5000.0):
    """Run simulation with heterogeneous parameters for realistic dynamics"""
    
    nest.ResetKernel()
    nest.set_verbosity("M_WARNING")
    nest.resolution = 0.1  # Better time resolution

    # Populations
    N_E = int(N_total * frac_exc)
    N_I = N_total - N_E

    E_pop = nest.Create("iaf_cond_exp", N_E)
    I_pop = nest.Create("iaf_cond_exp", N_I)

    # Add heterogeneity - CRITICAL for realistic dynamics
    np.random.seed(42 + int(g_ratio * 10))  # Different seed for each condition
    
    # Randomize initial membrane potentials
    E_pop.V_m = -70.0 + np.random.randn(N_E) * 5.0
    I_pop.V_m = -70.0 + np.random.randn(N_I) * 5.0
    
    # Slightly heterogeneous thresholds
    E_pop.V_th = -50.0 + np.random.randn(N_E) * 2.0
    I_pop.V_th = -50.0 + np.random.randn(N_I) * 2.0
    
    E_pop.t_ref = 2.0
    I_pop.t_ref = 1.0
    
    # Different time constants for E and I
    E_pop.tau_syn_ex = 2.0
    E_pop.tau_syn_in = 8.0  # Slower inhibition
    I_pop.tau_syn_ex = 1.0
    I_pop.tau_syn_in = 4.0

    # External drive with heterogeneity
    ext_drives = []
    for i in range(N_total):
        pg = nest.Create("poisson_generator")
        # Add variability to external input
        pg.rate = nu_ext * 1000.0 * (1.0 + np.random.randn() * 0.2)
        ext_drives.append(pg)
    
    # Background noise
    noise = nest.Create("poisson_generator")
    noise.rate = 200.0

    # Synaptic weights - KEY: scale with g_ratio
    g_E_base = 5.0
    delay = 1.5
    
    # Connect external drives individually
    for i, pg in enumerate(ext_drives):
        if i < N_E:
            nest.Connect(pg, E_pop[i:i+1], syn_spec={"weight": g_E_base, "delay": delay})
        else:
            nest.Connect(pg, I_pop[i-N_E:i-N_E+1], syn_spec={"weight": g_E_base, "delay": delay})
    
    # Background noise
    nest.Connect(noise, E_pop, "all_to_all", {"weight": g_E_base * 0.3, "delay": delay})
    nest.Connect(noise, I_pop, "all_to_all", {"weight": g_E_base * 0.3, "delay": delay})

    # Recurrent connections - THIS IS WHERE g_ratio MATTERS
    g_E = g_E_base
    g_I = g_ratio * g_E  # Higher g_ratio = stronger inhibition

    conn_spec = {"rule": "pairwise_bernoulli", "p": p_conn}

    print(f"    g_E = {g_E:.2f}, g_I = {g_I:.2f} (ratio = {g_ratio:.2f})")

    # Fixed delay for recurrent connections
    nest.Connect(E_pop, E_pop, conn_spec, {"weight": g_E, "delay": delay})
    nest.Connect(E_pop, I_pop, conn_spec, {"weight": g_E * 1.2, "delay": delay})
    nest.Connect(I_pop, E_pop, conn_spec, {"weight": -g_I, "delay": delay})
    nest.Connect(I_pop, I_pop, conn_spec, {"weight": -g_I * 0.8, "delay": delay})

    # Record from population
    mE = nest.Create("multimeter")
    mE.set(record_from=["V_m"], interval=1.0)
    n_record = min(30, N_E)
    nest.Connect(mE, E_pop[:n_record])
    
    # Also record spikes to check activity
    spike_rec = nest.Create("spike_recorder")
    nest.Connect(E_pop, spike_rec)

    # Run simulation with warmup
    warmup = 1000.0
    print(f"    Warming up for {warmup}ms...")
    nest.Simulate(warmup)
    
    print(f"    Recording for {sim_time}ms...")
    nest.Simulate(sim_time)

    # Check spiking activity
    spikes = spike_rec.get("events")
    n_spikes = len(spikes["times"])
    spike_rate = n_spikes / (N_E * sim_time / 1000.0)
    print(f"    Spike rate: {spike_rate:.2f} Hz/neuron, total spikes: {n_spikes}")

    # Analysis
    d = mE.get("events")
    
    if "times" in d and len(d["times"]) > 1000:
        t = np.array(d["times"])
        V_m = np.array(d["V_m"])
        senders = np.array(d["senders"])
        
        # Only use data after warmup
        warmup_mask = t > warmup
        t = t[warmup_mask]
        V_m = V_m[warmup_mask]
        senders = senders[warmup_mask]
        
        # Average across neurons (LFP proxy)
        unique_senders = np.unique(senders)
        V_m_traces = []
        for sender in unique_senders:
            mask = senders == sender
            V_m_traces.append(V_m[mask])
        
        V_m_avg = np.mean(V_m_traces, axis=0)
        
        print(f"    V_m stats: mean={V_m_avg.mean():.2f}, std={V_m_avg.std():.2f}")
        
        # Compute power spectrum
        fs = 1000.0  # 1 kHz sampling
        
        # Use longer segments for smoother spectrum
        nperseg = min(4096, len(V_m_avg)//4)
        f, Pxx = welch(V_m_avg, fs=fs, nperseg=nperseg, noverlap=nperseg*3//4)
        
        # Focus on 1-40 Hz
        freq_mask = (f >= 1) & (f <= 40)
        f_filt = f[freq_mask]
        Pxx_filt = Pxx[freq_mask]
        
        if len(f_filt) > 5:
            # Normalize
            Pxx_rel = Pxx_filt / np.sum(Pxx_filt)
            
            print(f"    Spectrum: {len(f_filt)} points, Pxx_rel range: [{Pxx_rel.min():.4f}, {Pxx_rel.max():.4f}]")
            print(f"    ✓ Success")
            
            return {
                'condition': condition_name,
                'g_ratio': g_ratio, 
                'f': f_filt, 
                'Pxx': Pxx_rel,
                'spike_rate': spike_rate,
                'success': True
            }
        else:
            print(f"    ✗ Insufficient frequency points")
            return {'condition': condition_name, 'g_ratio': g_ratio, 'success': False}
    else:
        print(f"    ✗ Insufficient data")
        return {'condition': condition_name, 'g_ratio': g_ratio, 'success': False}

# ------------------------------
# Main execution
# ------------------------------

# Lower g_ratio = less inhibition (AD)
# Higher g_ratio = more inhibition (healthy)
conditions = [
    ("AD", 3.5),   # Reduced inhibition
    ("HC2", 5.0),  # Medium inhibition
    ("HC3", 6.5),  # Strong inhibition
]

all_spectra = []

print("Starting simulations")
print("="*60)

for condition_name, g_ratio in conditions:
    print(f"\n[{condition_name}] g_I/g_E = {g_ratio:.1f}")
    result = run_simulation(condition_name, g_ratio)
    if result['success']:
        all_spectra.append(result)
        print(f"  >> Added (total: {len(all_spectra)})")

print("\n" + "="*60)
print(f"Results: {len(all_spectra)}/{len(conditions)} successful")

# Show spike rates
if len(all_spectra) > 0:
    print("\nSpike rates:")
    for s in all_spectra:
        print(f"  {s['condition']}: {s['spike_rate']:.2f} Hz/neuron")

print("="*60)

# Plotting
if len(all_spectra) > 0:
    print(f"\nCreating plot...")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = {'AD': '#90EE90', 'HC3': '#FFB6C1', 'HC2': '#A9A9A9'}
    linestyles = {'AD': '--', 'HC2': '-', 'HC3': '-'}
    
    for spectrum in all_spectra:
        condition = spectrum['condition']
        print(f"  {condition}: Pxx range [{spectrum['Pxx'].min():.4f}, {spectrum['Pxx'].max():.4f}]")
        
        ax.plot(spectrum['f'], spectrum['Pxx'], 
               label=condition, 
               linewidth=2.5, 
               color=colors.get(condition, 'black'),
               linestyle=linestyles.get(condition, '-'),
               alpha=0.8)
    
    ax.set_xlabel("Frequency (Hz)", fontsize=14, fontweight='bold')
    ax.set_ylabel("Relative power", fontsize=14, fontweight='bold')
    ax.legend(fontsize=12, loc='upper right', framealpha=0.9)
    ax.set_xlim([1, 40])
    ax.grid(True, alpha=0.2)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    
    output_file = 'power_spectra_AD_vs_HC_final.png'
    fig.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved: {output_file}")
    
    output_pdf = 'power_spectra_AD_vs_HC_final.pdf'
    fig.savefig(output_pdf, bbox_inches='tight')
    print(f"✓ Saved: {output_pdf}")
else:
    print("\nERROR: No successful simulations!")

print("\nDone!")
