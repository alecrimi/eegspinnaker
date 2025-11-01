# ============================================================
# PyNN-based simulation for EBRAINS neuromorphic hardware
# Compatible with BrainScaleS and SpiNNaker
# E/I imbalance in Alzheimer's disease
# Martinez-Cañada et al. 2023 replication
# ============================================================

#import pyNN.nest as sim  # Use pyNN.spiNNaker for SpiNNaker hardware
import pyNN.spiNNaker as sim  # Uncomment for SpiNNaker
# import pyNN.hardware.brainscales2 as sim  # Uncomment for BrainScaleS-2

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch
import os

def run_simulation(condition_name, g_ratio, N_total=1000, frac_exc=0.8, p_conn=0.15, 
                   nu_ext=5.0, sim_time=5000.0, timestep=0.1):
    """
    Run simulation using PyNN API for neuromorphic hardware compatibility
    
    Note: Network size reduced for hardware constraints
    """
    
    # Setup simulator
    sim.setup(timestep=timestep, min_delay=1.0, max_delay=10.0)
    
    # Population sizes
    N_E = int(N_total * frac_exc)
    N_I = N_total - N_E
    
    print(f"    Building network: {N_E} E-cells, {N_I} I-cells")
    
    # Neuron parameters - using standard PyNN conductance-based model
    cell_params = {
        'cm': 1.0,          # membrane capacitance (nF)
        'tau_m': 20.0,      # membrane time constant (ms)
        'tau_syn_E': 2.0,   # excitatory synaptic time constant (ms)
        'tau_syn_I': 8.0,   # inhibitory synaptic time constant (ms)
        'v_rest': -70.0,    # resting potential (mV)
        'v_reset': -70.0,   # reset potential (mV)
        'v_thresh': -50.0,  # spike threshold (mV)
        'tau_refrac': 2.0,  # refractory period (ms)
        'i_offset': 0.0,    # DC input current (nA)
    }
    
    # Separate params for inhibitory cells (faster dynamics)
    cell_params_I = cell_params.copy()
    cell_params_I['tau_syn_E'] = 1.0
    cell_params_I['tau_syn_I'] = 4.0
    cell_params_I['tau_refrac'] = 1.0
    
    # Create populations
    E_pop = sim.Population(N_E, sim.IF_cond_exp(**cell_params), label="Excitatory")
    I_pop = sim.Population(N_I, sim.IF_cond_exp(**cell_params_I), label="Inhibitory")
    
    # Initialize membrane potentials with heterogeneity
    np.random.seed(42 + int(g_ratio * 10))
    v_init_dist_E = sim.RandomDistribution('normal', mu=-70.0, sigma=5.0)
    v_init_dist_I = sim.RandomDistribution('normal', mu=-70.0, sigma=5.0)
    E_pop.initialize(v=v_init_dist_E)
    I_pop.initialize(v=v_init_dist_I)
    
    # External Poisson input
    print(f"    Creating external input sources...")
    ext_E = sim.Population(N_E, sim.SpikeSourcePoisson(rate=nu_ext * 1000.0), 
                           label="External_E")
    ext_I = sim.Population(N_I, sim.SpikeSourcePoisson(rate=nu_ext * 1000.0), 
                           label="External_I")
    
    # Background noise
    noise_source = sim.Population(1, sim.SpikeSourcePoisson(rate=200.0), 
                                  label="Noise")
    
    # Synaptic parameters
    g_E_base = 0.1  # nS - adjusted for PyNN units
    delay = 1.5     # ms
    
    print(f"    g_E = {g_E_base:.3f} nS, g_I/g_E ratio = {g_ratio:.2f}")
    
    # Synaptic weights (nS for conductance-based)
    w_ext = g_E_base
    w_E = g_E_base
    w_I = g_ratio * g_E_base  # Scaled by g_ratio
    
    # Connect external inputs (one-to-one)
    print(f"    Connecting external drives...")
    ext_E_conn = sim.Projection(ext_E, E_pop, 
                                 sim.OneToOneConnector(),
                                 synapse_type=sim.StaticSynapse(weight=w_ext, delay=delay),
                                 receptor_type='excitatory')
    
    ext_I_conn = sim.Projection(ext_I, I_pop,
                                 sim.OneToOneConnector(),
                                 synapse_type=sim.StaticSynapse(weight=w_ext, delay=delay),
                                 receptor_type='excitatory')
    
    # Background noise to all neurons
    noise_E = sim.Projection(noise_source, E_pop,
                             sim.AllToAllConnector(),
                             synapse_type=sim.StaticSynapse(weight=w_ext*0.3, delay=delay),
                             receptor_type='excitatory')
    
    noise_I = sim.Projection(noise_source, I_pop,
                             sim.AllToAllConnector(),
                             synapse_type=sim.StaticSynapse(weight=w_ext*0.3, delay=delay),
                             receptor_type='excitatory')
    
    # Recurrent connections - THIS IS WHERE g_ratio MATTERS
    print(f"    Creating recurrent connections...")
    
    # E -> E
    EE_conn = sim.Projection(E_pop, E_pop,
                             sim.FixedProbabilityConnector(p_connect=p_conn),
                             synapse_type=sim.StaticSynapse(weight=w_E, delay=delay),
                             receptor_type='excitatory')
    
    # E -> I
    EI_conn = sim.Projection(E_pop, I_pop,
                             sim.FixedProbabilityConnector(p_connect=p_conn),
                             synapse_type=sim.StaticSynapse(weight=w_E*1.2, delay=delay),
                             receptor_type='excitatory')
    
    # I -> E (inhibitory)
    IE_conn = sim.Projection(I_pop, E_pop,
                             sim.FixedProbabilityConnector(p_connect=p_conn),
                             synapse_type=sim.StaticSynapse(weight=w_I, delay=delay),
                             receptor_type='inhibitory')
    
    # I -> I (inhibitory)
    II_conn = sim.Projection(I_pop, I_pop,
                             sim.FixedProbabilityConnector(p_connect=p_conn),
                             synapse_type=sim.StaticSynapse(weight=w_I*0.8, delay=delay),
                             receptor_type='inhibitory')
    
    # Record voltage and spikes
    n_record = min(30, N_E)
    E_pop[0:n_record].record(['v'])
    E_pop.record(['spikes'])
    
    print(f"    Running simulation...")
    
    # Warmup period
    warmup = 1000.0
    print(f"      Warmup: {warmup} ms")
    sim.run(warmup)
    
    # Reset recordings after warmup
    E_pop.write_data('warmup_data.pkl', clear=True)
    
    # Main simulation
    print(f"      Recording: {sim_time} ms")
    sim.run(sim_time)
    
    # Extract data
    print(f"    Extracting data...")
    
    # Get voltage traces
    v_data = E_pop.get_data('v')
    v_segments = v_data.segments[0]
    
    # Get spikes
    spike_data = E_pop.get_data('spikes')
    spike_segments = spike_data.segments[0]
    
    # Calculate spike rate
    n_spikes = len(spike_segments.spiketrains[0])
    for st in spike_segments.spiketrains[1:]:
        n_spikes += len(st)
    spike_rate = n_spikes / (N_E * sim_time / 1000.0)
    
    print(f"    Spike rate: {spike_rate:.2f} Hz/neuron, total spikes: {n_spikes}")
    
    # Process voltage data
    success = False
    result = {'condition': condition_name, 'g_ratio': g_ratio, 'success': False}
    
    if len(v_segments.analogsignals) > 0:
        # Get voltage traces
        v_signal = v_segments.analogsignals[0]  # AnalogSignal object
        times = np.array(v_signal.times)
        voltages = np.array(v_signal)
        
        # Average across recorded neurons (LFP proxy)
        V_m_avg = np.mean(voltages, axis=1)
        
        print(f"    V_m stats: mean={V_m_avg.mean():.2f}, std={V_m_avg.std():.2f}")
        
        if len(V_m_avg) > 1000:
            # Compute power spectrum
            fs = 1000.0 / timestep  # Sampling frequency
            nperseg = min(4096, len(V_m_avg)//4)
            
            f, Pxx = welch(V_m_avg, fs=fs, nperseg=nperseg, noverlap=nperseg*3//4)
            
            # Focus on 1-40 Hz
            freq_mask = (f >= 1) & (f <= 40)
            f_filt = f[freq_mask]
            Pxx_filt = Pxx[freq_mask]
            
            if len(f_filt) > 5:
                # Normalize
                Pxx_rel = Pxx_filt / np.sum(Pxx_filt)
                
                print(f"    Spectrum: {len(f_filt)} points")
                print(f"    ✓ Success")
                
                result = {
                    'condition': condition_name,
                    'g_ratio': g_ratio,
                    'f': f_filt,
                    'Pxx': Pxx_rel,
                    'spike_rate': spike_rate,
                    'success': True
                }
                success = True
    
    if not success:
        print(f"    ✗ Insufficient data")
    
    # Clean up
    sim.end()
    
    return result

# ==============================================================================
# Main execution
# ==============================================================================

def main():
    """Main function for EBRAINS neuromorphic hardware execution"""
    
    # Create output directory
    output_dir = 'output'
    os.makedirs(output_dir, exist_ok=True)
    
    print("="*60)
    print("EBRAINS Neuromorphic Hardware Simulation")
    print("Using PyNN API for hardware compatibility")
    print("="*60)
    
    # Simulation conditions
    # Note: Smaller network due to hardware constraints
    conditions = [
        ("AD", 3.5),   # Reduced inhibition (Alzheimer's)
        ("HC2", 5.0),  # Medium inhibition
        ("HC3", 6.5),  # Strong inhibition (Healthy)
    ]
    
    all_spectra = []
    
    print("\nStarting simulations...")
    print("="*60)
    
    for condition_name, g_ratio in conditions:
        print(f"\n[{condition_name}] g_I/g_E = {g_ratio:.1f}")
        
        try:
            result = run_simulation(condition_name, g_ratio, 
                                   N_total=1000,  # Reduced for hardware
                                   sim_time=5000.0)
            if result['success']:
                all_spectra.append(result)
                print(f"  >> Added (total: {len(all_spectra)})")
        except Exception as e:
            print(f"  ✗ Error: {e}")
            continue
    
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
        
        # Save outputs
        output_png = os.path.join(output_dir, 'power_spectra_AD_vs_HC_neuromorphic.png')
        output_pdf = os.path.join(output_dir, 'power_spectra_AD_vs_HC_neuromorphic.pdf')
        
        fig.savefig(output_png, dpi=300, bbox_inches='tight')
        print(f"\n✓ Saved: {output_png}")
        
        fig.savefig(output_pdf, bbox_inches='tight')
        print(f"✓ Saved: {output_pdf}")
        
        plt.close(fig)
    else:
        print("\nERROR: No successful simulations!")
    
    print("\nDone!")
    return all_spectra

# Entry point
if __name__ == "__main__":
    results = main()
