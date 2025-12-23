# ============================================================
# PyNN-based simulation for EBRAINS SpiNNaker hardware
# FIXED VERSION based on error analysis
# E/I imbalance in Alzheimer's disease
# The name variable g_ratio and g_i need to be swapped
# ============================================================

import pyNN.spiNNaker as sim
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch
import os
#change neurons to 5000 and simtime to 50000
def run_simulation(condition_name, g_ratio, N_total=1000, frac_exc=0.8, p_conn=0.15, 
                   nu_ext=2.0, sim_time=10000.0, timestep=1.0):  # Increased simulation time
    """
    Run simulation optimized for SpiNNaker hardware - FIXED VERSION
    """
    
    # Setup simulator with SpiNNaker-specific parameters
    sim.setup(
        timestep=timestep, 
        min_delay=1.0, 
        max_delay=10.0,  # SpiNNaker ignores max_delay but keep for compatibility
        n_chips_required=None,
        n_boards_required=1
    )
    
    # Population sizes
    N_E = int(N_total * frac_exc)
    N_I = N_total - N_E
    
    print(f"    Building network: {N_E} E-cells, {N_I} I-cells")
    
    # Neuron parameters - adjusted for better dynamics
    cell_params = {
        'cm': 1.0,          # membrane capacitance (nF)
        'tau_m': 20.0,      # membrane time constant (ms)
        'tau_syn_E': 2.0,   # excitatory synaptic time constant (ms)
        'tau_syn_I': 8.0,   # inhibitory synaptic time constant (ms)
        'v_rest': -65.0,    # resting potential (mV) - adjusted
        'v_reset': -65.0,   # reset potential (mV)
        'v_thresh': -50.0,  # spike threshold (mV)
        'tau_refrac': 2.0,  # refractory period (ms)
        'i_offset': 0.0,    # DC input current (nA)
    }
    
    # Separate params for inhibitory cells
    cell_params_I = cell_params.copy()
    cell_params_I['tau_syn_E'] = 1.0
    cell_params_I['tau_syn_I'] = 4.0
    cell_params_I['tau_refrac'] = 1.0
    
    # Create populations
    E_pop = sim.Population(N_E, sim.IF_cond_exp(**cell_params), label="Excitatory")
    I_pop = sim.Population(N_I, sim.IF_cond_exp(**cell_params_I), label="Inhibitory")
    
    # Initialize membrane potentials with heterogeneity
    np.random.seed(42 + int(g_ratio * 10))
    E_pop.initialize(v=sim.RandomDistribution('normal', mu=-65.0, sigma=5.0))
    I_pop.initialize(v=sim.RandomDistribution('normal', mu=-65.0, sigma=5.0))
    
    # External Poisson input - REDUCED rates for stability
    print(f"    Creating external input sources...")
    ext_rate = nu_ext * 50.0  # SIGNIFICANTLY REDUCED for SpiNNaker
    ext_E = sim.Population(N_E, sim.SpikeSourcePoisson(rate=ext_rate), 
                           label="External_E")
    ext_I = sim.Population(N_I, sim.SpikeSourcePoisson(rate=ext_rate), 
                           label="External_I")
    
    # Background noise - REDUCED
    noise_source = sim.Population(1, sim.SpikeSourcePoisson(rate=50.0), 
                                  label="Noise")
    
    # Synaptic parameters - FURTHER REDUCED for stability
    g_E_base = 0.02  # Further reduced for SpiNNaker stability
    delay = 2.0     # Use 2.0ms directly to avoid rounding warnings
    
    print(f"    g_E = {g_E_base:.3f} nS, g_I/g_E ratio = {g_ratio:.2f}")
    
    # Synaptic weights
    w_ext = g_E_base
    w_E = g_E_base
    w_I = g_ratio * g_E_base
    
    # Connect external inputs
    print(f"    Connecting external drives...")
    sim.Projection(ext_E, E_pop, 
                   sim.OneToOneConnector(),
                   sim.StaticSynapse(weight=w_ext, delay=delay),
                   receptor_type='excitatory')
    
    sim.Projection(ext_I, I_pop,
                   sim.OneToOneConnector(),
                   sim.StaticSynapse(weight=w_ext, delay=delay),
                   receptor_type='excitatory')
    
    # Background noise
    sim.Projection(noise_source, E_pop,
                   sim.AllToAllConnector(),
                   sim.StaticSynapse(weight=w_ext*0.3, delay=delay),
                   receptor_type='excitatory')
    
    sim.Projection(noise_source, I_pop,
                   sim.AllToAllConnector(),
                   sim.StaticSynapse(weight=w_ext*0.3, delay=delay),
                   receptor_type='excitatory')
    
    # Recurrent connections
    print(f"    Creating recurrent connections...")
    
    # Use lower connection probability for stability
    p_conn_recurrent = 0.1  # Reduced from 0.15
    
    sim.Projection(E_pop, E_pop,
                   sim.FixedProbabilityConnector(p_conn_recurrent),
                   sim.StaticSynapse(weight=w_E, delay=delay),
                   receptor_type='excitatory')
    
    sim.Projection(E_pop, I_pop,
                   sim.FixedProbabilityConnector(p_conn_recurrent),
                   sim.StaticSynapse(weight=w_E*1.2, delay=delay),
                   receptor_type='excitatory')
    
    sim.Projection(I_pop, E_pop,
                   sim.FixedProbabilityConnector(p_conn_recurrent),
                   sim.StaticSynapse(weight=-w_I, delay=delay),
                   receptor_type='inhibitory')
    
    sim.Projection(I_pop, I_pop,
                   sim.FixedProbabilityConnector(p_conn_recurrent),
                   sim.StaticSynapse(weight=-w_I*0.8, delay=delay),
                   receptor_type='inhibitory')
    
    # Record data - INCREASE recorded neurons for better LFP
    n_record = min(50, N_E)  # Increased from 10 to 50
    E_pop[0:n_record].record(['v'])
    E_pop.record(['spikes'])
    I_pop.record(['spikes'])
    
    print(f"    Running simulation for {sim_time}ms...")
    
    # Run simulation - no separate warmup needed
    sim.run(sim_time)
    
    # Extract data
    print(f"    Extracting data...")
    
    try:
        # Get data
        v_data = E_pop.get_data('v')
        spike_data_E = E_pop.get_data('spikes')
        spike_data_I = I_pop.get_data('spikes')
        
        success = False
        result = {'condition': condition_name, 'g_ratio': g_ratio, 'success': False}
        
        # Check if we have voltage data
        if v_data.segments and len(v_data.segments[0].analogsignals) > 0:
            v_signal = v_data.segments[0].analogsignals[0]
            times = np.array(v_signal.times)
            voltages = np.array(v_signal)
            
            print(f"    Voltage data shape: {voltages.shape}")
            
            # Average across recorded neurons (LFP proxy)
            V_m_avg = np.mean(voltages, axis=1)
            
            print(f"    V_m stats: mean={V_m_avg.mean():.2f}, std={V_m_avg.std():.2f}")
            print(f"    V_m length: {len(V_m_avg)} samples")
            
            # Calculate spike rates
            n_spikes_E = sum(len(st) for st in spike_data_E.segments[0].spiketrains)
            n_spikes_I = sum(len(st) for st in spike_data_I.segments[0].spiketrains)
            spike_rate_E = n_spikes_E / (N_E * sim_time / 1000.0)
            spike_rate_I = n_spikes_I / (N_I * sim_time / 1000.0)
            
            print(f"    Spike rates: E={spike_rate_E:.2f} Hz, I={spike_rate_I:.2f} Hz")
            
            # Check if we have enough data for spectral analysis
            if len(V_m_avg) > 2000:  # Reduced threshold
                # Compute power spectrum
                fs = 1000.0 / timestep  # 1000 Hz sampling
                nperseg = min(2048, len(V_m_avg)//4)
                
                print(f"    Computing PSD with nperseg={nperseg}")
                
                f, Pxx = welch(V_m_avg, fs=fs, nperseg=nperseg, noverlap=nperseg//2)
                
                # Focus on 1-40 Hz
                freq_mask = (f >= 1) & (f <= 40)
                f_filt = f[freq_mask]
                Pxx_filt = Pxx[freq_mask]
                
                if len(f_filt) > 5:
                    # Normalize
                    Pxx_rel = Pxx_filt / np.sum(Pxx_filt)
                    
                    print(f"    Spectrum computed: {len(f_filt)} frequency points")
                    print(f"    ✓ Success - condition {condition_name} completed")
                    
                    result = {
                        'condition': condition_name,
                        'g_ratio': g_ratio,
                        'ge_gi_ratio': 1.0/g_ratio,  # Add g_E/g_I ratio
                        'f': f_filt,
                        'Pxx': Pxx_rel,
                        'spike_rate_E': spike_rate_E,
                        'spike_rate_I': spike_rate_I,
                        'v_mean': V_m_avg.mean(),
                        'v_std': V_m_avg.std(),
                        'success': True
                    }
                    success = True
                else:
                    print(f"    ✗ Not enough frequency points after filtering: {len(f_filt)}")
            else:
                print(f"    ✗ Insufficient voltage data length: {len(V_m_avg)} samples")
        else:
            print(f"    ✗ No voltage data available")
        
        if not success:
            print(f"    ✗ Data processing failed for {condition_name}")
        
        return result
        
    except Exception as e:
        print(f"    ✗ Data extraction error: {e}")
        import traceback
        traceback.print_exc()
        return {'condition': condition_name, 'g_ratio': g_ratio, 'success': False}
    
    finally:
        # Always clean up
        sim.end()

def calculate_1f_slope(frequencies, power_spectrum):
    """Calculate 1/f slope from power spectrum"""
    try:
        # Use only frequencies > 0 and power > 0
        mask = (frequencies > 1) & (power_spectrum > 0) & (frequencies < 20)  # Limit to lower frequencies
        if np.sum(mask) < 5:
            return 0.0, False
            
        f_log = np.log10(frequencies[mask])
        p_log = np.log10(power_spectrum[mask])
        
        # Linear fit
        slope, intercept = np.polyfit(f_log, p_log, 1)
        return slope, True
    except Exception as e:
        print(f"    1/f slope calculation error: {e}")
        return 0.0, False

# ==============================================================================
# Main execution
# ==============================================================================

def main():
    """Main function for EBRAINS SpiNNaker execution"""
    
    # Create output directory
    output_dir = 'output'
    os.makedirs(output_dir, exist_ok=True)
    
    print("="*60)
    print("EBRAINS SpiNNaker Simulation - FIXED VERSION")
    print("AD vs HC E/I Balance Analysis")
    print("="*60)
    
    # Simulation conditions - test with wider range
    conditions = [
        ("AD", 2.5),   # Very reduced inhibition
        ("Mild_AD", 3.5),   
        ("HC", 6.5),   # Healthy control
        ("Strong_HC", 8.5), # Very strong inhibition
    ]
    
    all_results = []
    
    print("\nStarting FIXED SpiNNaker simulations...")
    print("="*60)
    
    for condition_name, g_ratio in conditions:
        print(f"\n[{condition_name}] g_I/g_E = {g_ratio:.1f}")
        
        try:
            result = run_simulation(
                condition_name, 
                g_ratio, 
                N_total=500,       # Keep small for testing
                sim_time=10000.0,  # INCREASED simulation time
                nu_ext=1.0,        # REDUCED external input
                timestep=1.0
            )
            
            if result['success']:
                # Calculate 1/f slope
                slope, slope_success = calculate_1f_slope(result['f'], result['Pxx'])
                if slope_success:
                    result['slope_1f'] = slope
                    print(f"    1/f slope: {slope:.4f}")
                
                all_results.append(result)
                print(f"  >> Successfully added (total: {len(all_results)})")
            else:
                print(f"  ✗ Simulation failed")
                
        except Exception as e:
            print(f"  ✗ Fatal error: {e}")
            continue
    
    print("\n" + "="*60)
    print(f"Results: {len(all_results)}/{len(conditions)} successful")
    
    # Analysis and plotting
    if len(all_results) > 0:
        print("\nSummary of successful simulations:")
        for r in all_results:
            print(f"  {r['condition']}: g_E/g_I={r.get('ge_gi_ratio', 0):.3f}, "
                  f"E-rate={r['spike_rate_E']:.2f}Hz, I-rate={r['spike_rate_I']:.2f}Hz")
        
        # Create comprehensive plots
        create_analysis_plots(all_results, output_dir)
        
    else:
        print("\nERROR: No successful simulations!")
    
    print("\nDone!")
    return all_results

def create_analysis_plots(results, output_dir):
    """Create analysis plots from successful simulations"""
    
    # Plot 1: Power spectra comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Color map for conditions
    colors = plt.cm.viridis(np.linspace(0, 1, len(results)))
    
    # Plot power spectra
    for i, r in enumerate(results):
        ax1.plot(r['f'], r['Pxx'], 
                label=f"{r['condition']} (g_E/g_I={1.0/r['g_ratio']:.2f})",
                color=colors[i], linewidth=2, alpha=0.8)
    
    ax1.set_xlabel("Frequency (Hz)", fontsize=12)
    ax1.set_ylabel("Relative Power", fontsize=12)
    ax1.set_title("Power Spectra - SpiNNaker", fontsize=14)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(1, 40)
    
    # Plot g_E/g_I ratio vs spike rates
    ge_gi_ratios = [1.0/r['g_ratio'] for r in results]
    e_rates = [r['spike_rate_E'] for r in results]
    i_rates = [r['spike_rate_I'] for r in results]
    
    ax2.scatter(ge_gi_ratios, e_rates, c=colors, s=100, alpha=0.7, label='Excitatory')
    ax2.scatter(ge_gi_ratios, i_rates, c=colors, s=100, alpha=0.7, marker='s', label='Inhibitory')
    
    # Add labels for each point
    for i, r in enumerate(results):
        ax2.annotate(r['condition'], (ge_gi_ratios[i], e_rates[i]), 
                    xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    ax2.set_xlabel("g_E / g_I Ratio", fontsize=12)
    ax2.set_ylabel("Firing Rate (Hz)", fontsize=12)
    ax2.set_title("E/I Balance vs Firing Rates", fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plots
    output_file = os.path.join(output_dir, 'spinnaker_analysis_fixed.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Saved analysis plot: {output_file}")
    
    # Save results data
    import pickle
    data_file = os.path.join(output_dir, 'spinnaker_results.pkl')
    with open(data_file, 'wb') as f:
        pickle.dump(results, f)
    print(f"✓ Saved results data: {data_file}")

if __name__ == "__main__":
    results = main()
