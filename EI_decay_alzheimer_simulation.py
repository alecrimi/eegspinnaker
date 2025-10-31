# ============================================================
# 
# PyNEST prototype - scalable from laptop to HPC to EBRAINS
# E/I decay duet to Alzheimer
# ============================================================

import nest
import numpy as np
import matplotlib.pyplot as plt

# ------------------------------
# Simulation parameters
# ------------------------------
N_total = 100            # try 5000 later
frac_exc = 0.8
p_conn = 0.2
nu_ext = 3.0             # Hz (external Poisson drive)
sim_time = 1000.0        # ms

# sweep of inhibitory/excitatory conductance ratio g = g_I/g_E
g_ratios = [0.06, 0.1, 0.15, 0.2]

# ------------------------------
# Neuron and synapse parameters
# ------------------------------
neuron_params = {
    "C_m": 200.0,       # pF
    "tau_m": 20.0,      # ms
    "t_ref": 2.0,       # ms
    "E_L": -70.0,       # mV
    "V_th": -50.0,      # mV
    "V_reset": -60.0,   # mV
    "tau_syn_ex": 5.0,  # ms (AMPA)
    "tau_syn_in": 10.0  # ms (GABA)
}

# Synaptic strengths (baseline excitatory)
g_E_base = 4.0          # nS (arbitrary scaling)
delay = 1.5             # ms

# ------------------------------
# Loop over E/I ratios
# ------------------------------
for g_ratio in g_ratios:

    nest.ResetKernel()
    nest.SetKernelStatus({"print_time": True})

    # Populations
    N_E = int(N_total * frac_exc)
    N_I = N_total - N_E

    E_pop = nest.Create("iaf_cond_exp", N_E, params=neuron_params)
    I_pop = nest.Create("iaf_cond_exp", N_I, params=neuron_params)

    # Poisson input: external (ν0 = 3 Hz) + noise (higher)
    ext_drive = nest.Create("poisson_generator", params={"rate": nu_ext})
    noise_drive = nest.Create("poisson_generator", params={"rate": 100.0})

    nest.Connect(ext_drive, E_pop, syn_spec={"weight": g_E_base, "delay": delay})
    nest.Connect(ext_drive, I_pop, syn_spec={"weight": g_E_base, "delay": delay})
    nest.Connect(noise_drive, E_pop, syn_spec={"weight": g_E_base, "delay": delay})
    nest.Connect(noise_drive, I_pop, syn_spec={"weight": g_E_base, "delay": delay})

    # ------------------------------
    # Recurrent connections
    # ------------------------------
    g_E = g_E_base
    g_I = g_ratio * g_E

    conn_dict = {"rule": "pairwise_bernoulli", "p": p_conn}

    nest.Connect(E_pop, E_pop, conn_dict, syn_spec={"weight": g_E, "delay": delay})
    nest.Connect(E_pop, I_pop, conn_dict, syn_spec={"weight": g_E, "delay": delay})
    nest.Connect(I_pop, E_pop, conn_dict, syn_spec={"weight": -g_I, "delay": delay})
    nest.Connect(I_pop, I_pop, conn_dict, syn_spec={"weight": -g_I, "delay": delay})

    # ------------------------------
    # Recordings
    # ------------------------------
    mE = nest.Create("multimeter", params={
        "withtime": True,
        "record_from": ["V_m", "I_syn_ex", "I_syn_in"]
    })
    nest.Connect(mE, E_pop[:5])  # record a few excitatory cells

    spE = nest.Create("spike_recorder")
    spI = nest.Create("spike_recorder")
    nest.Connect(E_pop, spE)
    nest.Connect(I_pop, spI)

    # ------------------------------
    # Run simulation
    # ------------------------------
    nest.Simulate(sim_time)

    # ------------------------------
    # Compute field potential proxy
    # ------------------------------
    d = nest.GetStatus(mE)[0]
    I_ex = np.array(d["events"]["I_syn_ex"])
    I_in = np.array(d["events"]["I_syn_in"])
    t = np.array(d["events"]["times"])
    # simple sum of absolute AMPA+GABA currents
    pseudo_field = np.abs(I_ex) + np.abs(I_in)

    # power spectrum (approx 1/f slope later)
    from scipy.signal import welch
    f, Pxx = welch(pseudo_field, fs=1000.0 / np.mean(np.diff(t)), nperseg=512)

    plt.loglog(f, Pxx, label=f"gI/gE={g_ratio:.2f}")

# ------------------------------
# Plot results
# ------------------------------
plt.xlabel("Frequency (Hz)")
plt.ylabel("Power")
plt.legend()
plt.title("Power spectra for different E/I ratios")
plt.show()
