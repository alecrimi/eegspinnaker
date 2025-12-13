# ============================================================
# PyNEST prototype – EEG/MEG‑comparable E/I imbalance model
# Inspired by Martínez‑Cañada et al. :
#   • EEG proxy based on synaptic currents (not Vm)
#   • Explicit separation of aperiodic (1/f) component
#   • Relative power computed AFTER removing 1/f
#   • Outputs directly comparable to Fig. 3–4 in the paper
# ============================================================

import nest
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.signal import welch
from numpy.polynomial.polynomial import polyfit

# ------------------------------------------------------------
# Utility: estimate aperiodic 1/f slope (log–log fit)
# ------------------------------------------------------------

def estimate_aperiodic_slope(f, Pxx, fmin=2.0, fmax=40.0):
    mask = (f >= fmin) & (f <= fmax)
    logf = np.log10(f[mask])
    logP = np.log10(Pxx[mask])
    b, a = polyfit(logf, logP, 1)  # logP ≈ a + b logf
    return -b, a  # slope χ > 0


# ------------------------------------------------------------
# Main simulation
# ------------------------------------------------------------

def run_simulation(condition_name,
                   g_ratio,
                   N_total=200,
                   frac_exc=0.8,
                   p_conn=0.15,
                   nu_ext=5.0,
                   sim_time=5000.0,
                   warmup=1000.0,
                   seed_base=42):

    nest.ResetKernel()
    nest.set_verbosity("M_WARNING")
    nest.resolution = 0.1

    rng_seed = seed_base + int(10 * g_ratio)
    np.random.seed(rng_seed)

    # --------------------
    # Populations
    # --------------------
    N_E = int(N_total * frac_exc)
    N_I = N_total - N_E

    E = nest.Create("iaf_cond_exp", N_E)
    I = nest.Create("iaf_cond_exp", N_I)

    # --------------------
    # Heterogeneity
    # --------------------
    for pop, N in [(E, N_E), (I, N_I)]:
        pop.V_m = -70 + 5 * np.random.randn(N)
        pop.V_th = -50 + 2 * np.random.randn(N)

    E.t_ref, I.t_ref = 2.0, 1.0
    E.tau_syn_ex, E.tau_syn_in = 2.0, 8.0
    I.tau_syn_ex, I.tau_syn_in = 1.0, 4.0

    # --------------------
    # External drive
    # --------------------
    g_E = 3.0
    g_I = g_ratio * g_E
    delay = 1.5

    for i in range(N_total):
        pg = nest.Create("poisson_generator")
        pg.rate = nu_ext * 1000 * np.random.lognormal(0, 0.2)
        if i < N_E:
            nest.Connect(pg, E[i:i+1], syn_spec={"weight": g_E, "delay": delay})
        else:
            nest.Connect(pg, I[i-N_E:i-N_E+1], syn_spec={"weight": g_E, "delay": delay})

    noise = nest.Create("poisson_generator")
    noise.rate = 200.0
    nest.Connect(noise, E, syn_spec={"weight": 0.3 * g_E, "delay": delay})
    nest.Connect(noise, I, syn_spec={"weight": 0.3 * g_E, "delay": delay})

    # --------------------
    # Recurrent connectivity
    # --------------------
    conn = {"rule": "pairwise_bernoulli", "p": p_conn}

    nest.Connect(E, E, conn, syn_spec={"weight": g_E, "delay": delay})
    nest.Connect(E, I, conn, syn_spec={"weight": 1.2 * g_E, "delay": delay})
    nest.Connect(I, E, conn, syn_spec={"weight": -g_I, "delay": delay})
    nest.Connect(I, I, conn, syn_spec={"weight": -0.8 * g_I, "delay": delay})

    # --------------------
    # Recording synaptic currents (EEG proxy)
    # --------------------
    mm = nest.Create("multimeter")
    mm.set(record_from=["I_syn_ex", "I_syn_in"], interval=1.0)
    nest.Connect(mm, E[:20])

    spike_rec = nest.Create("spike_recorder")
    nest.Connect(E, spike_rec)

    # --------------------
    # Simulate
    # --------------------
    nest.Simulate(warmup)
    nest.Simulate(sim_time)

    # --------------------
    # EEG proxy construction
    # --------------------
    ev = mm.get("events")
    t = ev["times"]
    mask = t > warmup

    I_ex = ev["I_syn_ex"][mask]
    I_in = ev["I_syn_in"][mask]

    # Martínez‑Cañada EEG proxy: weighted synaptic currents
    eeg_proxy = I_ex - I_in

    if len(eeg_proxy) < 2000:
        return {'success': False}

    fs = 1000.0 / mm.interval
    f, Pxx = welch(eeg_proxy, fs=fs, nperseg=2048, noverlap=1536)

    band = (f >= 1) & (f <= 40)
    f, Pxx = f[band], Pxx[band]

    # --------------------
    # Aperiodic / periodic separation
    # --------------------
    slope, offset = estimate_aperiodic_slope(f, Pxx)
    P_aperiodic = 10 ** (offset) * f ** (-slope)
    P_periodic = Pxx / P_aperiodic
    P_rel = P_periodic / np.sum(P_periodic)

    # --------------------
    # Spike rate
    # --------------------
    spikes = spike_rec.get("events")["times"]
    spike_rate = np.sum(spikes > warmup) / (N_E * sim_time / 1000)

    return {
        'condition': condition_name,
        'g_ratio': g_ratio,
        'f': f,
        'Pxx_rel': P_rel,
        'slope': slope,
        'spike_rate': spike_rate,
        'success': True
    }


# ============================================================
# Main execution
# ============================================================

conditions = [
    ("AD", 2.5),
    ("MCI", 3.5),
    ("HC", 6.5),
    ("HC strong", 8.5),
]

results = []

for name, g in conditions:
    print(f"Running {name} (g_I/g_E={g})")
    out = run_simulation(name, g)
    if out.get('success'):
        results.append(out)

# --------------------
# Plot (Martínez‑Cañada‑style relative power)
# --------------------
if results:
    fig, ax = plt.subplots(figsize=(10, 6))

    for r in results:
        ax.plot(r['f'], r['Pxx_rel'], label=f"{r['condition']} (χ={r['slope']:.2f})", lw=2.5)

    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Relative periodic power")
    ax.set_xlim(1, 40)
    ax.legend()
    ax.grid(alpha=0.2)

    plt.tight_layout()
    plt.savefig("EEG_proxy_EI_MartinezCanada_style.png", dpi=300)

print("Done.")
