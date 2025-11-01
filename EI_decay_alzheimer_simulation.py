# ============================================================
# 
# PyNEST prototype - scalable from laptop to HPC to EBRAINS
# E/I decay due to Alzheimer
# ============================================================

import nest
import numpy as np
import matplotlib.pyplot as plt

# ------------------------------
# Simulation parameters
# ------------------------------
N_total = 100
frac_exc = 0.8
p_conn = 0.2
nu_ext = 3.0
sim_time = 1000.0

g_ratios = [0.06, 0.1, 0.15, 0.2]

g_E_base = 4.0
delay = 1.5

# ------------------------------
# Loop over E/I ratios
# ------------------------------
for g_ratio in g_ratios:
    print(f"Simulating with g_ratio = {g_ratio}")

    nest.ResetKernel()
    nest.set_verbosity("M_WARNING")

    # Populations - use default parameters
    N_E = int(N_total * frac_exc)
    N_I = N_total - N_E

    E_pop = nest.Create("iaf_cond_exp", N_E)
    I_pop = nest.Create("iaf_cond_exp", N_I)

    # Set only the most essential parameters individually
    E_pop.V_m = -70.0
    I_pop.V_m = -70.0
    E_pop.V_th = -50.0
    I_pop.V_th = -50.0

    # Poisson input
    ext_drive = nest.Create("poisson_generator")
    ext_drive.rate = nu_ext
    
    noise_drive = nest.Create("poisson_generator")
    noise_drive.rate = 100.0

    # Connections
    syn_spec_ext = {"weight": g_E_base, "delay": delay}
    
    nest.Connect(ext_drive, E_pop, "all_to_all", syn_spec_ext)
    nest.Connect(ext_drive, I_pop, "all_to_all", syn_spec_ext)
    nest.Connect(noise_drive, E_pop, "all_to_all", syn_spec_ext)
    nest.Connect(noise_drive, I_pop, "all_to_all", syn_spec_ext)

    # Recurrent connections
    g_E = g_E_base
    g_I = g_ratio * g_E

    conn_spec = {"rule": "pairwise_bernoulli", "p": p_conn}

    nest.Connect(E_pop, E_pop, conn_spec, {"weight": g_E, "delay": delay})
    nest.Connect(E_pop, I_pop, conn_spec, {"weight": g_E, "delay": delay})
    nest.Connect(I_pop, E_pop, conn_spec, {"weight": -g_I, "delay": delay})
    nest.Connect(I_pop, I_pop, conn_spec, {"weight": -g_I, "delay": delay})

    # Recordings - simplest approach
    mE = nest.Create("multimeter")
    mE.set(record_from=["V_m"])
    nest.Connect(mE, E_pop[:3])

    spE = nest.Create("spike_recorder")
    spI = nest.Create("spike_recorder")
    nest.Connect(E_pop, spE)
    nest.Connect(I_pop, spI)

    # Run simulation
    nest.Simulate(sim_time)

    # Analysis - simplified
    d = mE.get("events")
    if "times" in d and len(d["times"]) > 1:
        t = np.array(d["times"])
        V_m = np.array(d["V_m"])
        
        from scipy.signal import welch
        fs = 1000.0 / np.mean(np.diff(t))
        f, Pxx = welch(V_m, fs=fs, nperseg=min(256, len(V_m)//4))
        plt.loglog(f, Pxx, label=f"gI/gE={g_ratio:.2f}")
    else:
        print(f"Warning: Not enough data for g_ratio {g_ratio}")

plt.xlabel("Frequency (Hz)")
plt.ylabel("Power")
plt.legend()
plt.title("Power spectra for different E/I ratios")
plt.show()
