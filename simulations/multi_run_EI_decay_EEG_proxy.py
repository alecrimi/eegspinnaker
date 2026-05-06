# ============================================================
# PyNEST – E/I imbalance model (Conductance-based LIF) - SMOOTH
# EEG/MEG proxy from synaptic currents
# Multi-run averaged version: original single-run code looped N times
# ============================================================
import nest
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.signal import welch, savgol_filter


def run_simulation(condition, g_ratio, N_total=5000, frac_exc=0.8, p_conn=0.2,
                   nu_ext=1100, sim_time=10000.0, warmup=2000.0, seed=42,  #920
                   record_fraction=0.5, smooth_spectrum=True):
    """
    Original single-run simulation — unchanged from the single-run script.
    """
    nest.ResetKernel()
    nest.resolution = 0.1
    nest.set_verbosity("M_WARNING")

    np.random.seed(seed)
    nest.SetKernelStatus({"rng_seed": seed})

    N_E = int(frac_exc * N_total)
    N_I = N_total - N_E

    E = nest.Create("iaf_cond_exp", N_E)
    I = nest.Create("iaf_cond_exp", N_I)

    neuron_params = {
        "C_m": 250.0, "g_L": 16.67, "E_L": -70.0,
        "V_th": -50.0, "V_reset": -70.0, "t_ref": 2.0,
        "E_ex": 0.0, "E_in": -80.0,
        "tau_syn_ex": 2.0, "tau_syn_in": 8.0,
    }
    for pop in (E, I):
        pop.set(neuron_params)
        pop.V_m = -70.0 + 5.0 * np.random.randn(len(pop))

    g_E = 2
    g_I = g_ratio * g_E
    delay = 1.5

    print(f"\n[{condition}] g_I/g_E={g_ratio:.2f}, seed={seed}")

    ext = nest.Create("poisson_generator")
    ext.rate = nu_ext # *  1000.0
    nest.Connect(ext, E, syn_spec={"weight": g_E, "delay": delay})
    nest.Connect(ext, I, syn_spec={"weight": g_E, "delay": delay})

    conn = {"rule": "pairwise_bernoulli", "p": p_conn}
    nest.Connect(E, E, conn, syn_spec={"weight":  g_E, "delay": delay})
    nest.Connect(E, I, conn, syn_spec={"weight":  g_E, "delay": delay})
    nest.Connect(I, E, conn, syn_spec={"weight":  -g_I, "delay": delay})
    nest.Connect(I, I, conn, syn_spec={"weight":  -g_I, "delay": delay})

    n_rec = max(50, int(record_fraction * N_E))
    n_rec = min(n_rec, N_E)
    mm = nest.Create("multimeter")
    mm.set({"interval": 1.0, "record_from": ["g_ex", "g_in", "V_m"]})
    nest.Connect(mm, E[:n_rec])

    spike_rec = nest.Create("spike_recorder")
    nest.Connect(E, spike_rec)

    nest.Simulate(warmup)
    nest.Simulate(sim_time)

    ev      = mm.get("events")
    times   = np.array(ev["times"])
    senders = np.array(ev["senders"])
    g_ex    = np.array(ev["g_ex"])
    g_in    = np.array(ev["g_in"])
    V_m     = np.array(ev["V_m"])

    I_ex = g_ex * (V_m - 0.0)
    I_in = g_in * (V_m - (-80.0))

    mask             = times > warmup
    times_f          = times[mask]
    senders_f        = senders[mask]
    I_ex_f           = I_ex[mask]
    I_in_f           = I_in[mask]

    unique_times   = np.sort(np.unique(times_f))
    unique_neurons = np.sort(np.unique(senders_f))
    n_times        = len(unique_times)
    n_neurons      = len(unique_neurons)

    expected = n_times * n_neurons
    if len(times_f) != expected:
        I_ex_mat = np.zeros((n_times, n_neurons))
        I_in_mat = np.zeros((n_times, n_neurons))
        n2i = {nid: idx for idx, nid in enumerate(unique_neurons)}
        t2i = {t:   idx for idx, t   in enumerate(unique_times)}
        for k in range(len(times_f)):
            I_ex_mat[t2i[times_f[k]], n2i[senders_f[k]]] = I_ex_f[k]
            I_in_mat[t2i[times_f[k]], n2i[senders_f[k]]] = I_in_f[k]
    else:
        I_ex_mat = I_ex_f.reshape(n_times, n_neurons)
        I_in_mat = I_in_f.reshape(n_times, n_neurons)

    lfp  = I_ex_mat.mean(axis=1) - I_in_mat.mean(axis=1)
    lfp -= lfp.mean()

    spike_times = spike_rec.get("events")["times"]
    spike_times = spike_times[spike_times > warmup]
    firing_rate = len(spike_times) / (sim_time / 1000.0) / N_E
    print(f"  FR={firing_rate:.2f} Hz, LFP samples={len(lfp)}")

    fs = 1000.0
    
    nperseg  = min(len(lfp) // 2, 16384)
    nperseg  = max(nperseg, 4096)
    noverlap = int(15 * nperseg // 16)

    f, Pxx = welch(lfp, fs=fs, nperseg=nperseg, noverlap=noverlap,
                   window='hann', detrend='constant')
         
    band = (f >= 1) & (f <= 40)
    f    = f[band]
    Pxx  = Pxx[band]

    if smooth_spectrum:
        wl = min(21, len(Pxx) // 2 * 2 - 1)
        if wl >= 5: #5
            Pxx = savgol_filter(Pxx, window_length=wl, polyorder=3)

    Pxx = np.maximum(Pxx, 0)
    Pxx /= Pxx.sum()

    return {"condition": condition, "g_ratio": g_ratio,
            "f": f, "Pxx": Pxx, "firing_rate": firing_rate}


def find_working_params(N_total=400, sim_time=2000.0, warmup=1000.0, seed=42):
    """
    Quick sweep over ext.rate and g_ratio.
    Uses a small network (N=400) for speed.
    Target: FR between 5 and 20 Hz.
    """
    nu_values = [1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900]
    g_values  = [2.5, 3.5, 4.5, 5.5, 6.5]

    print("\n" + "="*55)
    print("PARAMETER SWEEP — target FR: 5–20 Hz")
    print(f"{'nu_ext':>8} | " + " ".join(f"g={g:.1f}" for g in g_values))
    print("-"*55)

    for nu in nu_values:
        row = f"{nu:>8} | "
        for g_ratio in g_values:
            nest.ResetKernel()
            nest.resolution = 0.1
            nest.set_verbosity("M_WARNING")
            nest.SetKernelStatus({"rng_seed": seed})

            N_E = int(0.8 * N_total)
            N_I = N_total - N_E

            E = nest.Create("iaf_cond_exp", N_E)
            I = nest.Create("iaf_cond_exp", N_I)

            params = {
                "C_m": 250.0, "g_L": 16.67, "E_L": -70.0,
                "V_th": -50.0, "V_reset": -70.0, "t_ref": 2.0,
                "E_ex": 0.0,   "E_in": -80.0,
                "tau_syn_ex": 2.0, "tau_syn_in": 8.0,
            }
            for pop in (E, I):
                pop.set(params)

            g_E = 2.0
            g_I = g_ratio * g_E
            delay = 1.5

            ext = nest.Create("poisson_generator")
            ext.rate = float(nu)
            nest.Connect(ext, E, syn_spec={"weight": g_E, "delay": delay})
            nest.Connect(ext, I, syn_spec={"weight": g_E, "delay": delay})

            conn = {"rule": "pairwise_bernoulli", "p": 0.2}
            nest.Connect(E, E, conn, syn_spec={"weight":  g_E, "delay": delay})
            nest.Connect(E, I, conn, syn_spec={"weight":  g_E, "delay": delay})
            nest.Connect(I, E, conn, syn_spec={"weight": -g_I, "delay": delay})
            nest.Connect(I, I, conn, syn_spec={"weight": -g_I, "delay": delay})

            spike_rec = nest.Create("spike_recorder")
            nest.Connect(E, spike_rec)

            nest.Simulate(warmup + sim_time)

            spike_times = spike_rec.get("events")["times"]
            spike_times = spike_times[spike_times > warmup]
            fr = len(spike_times) / (sim_time / 1000.0) / N_E

            tag = " ✓" if 5 <= fr <= 20 else ""
            row += f"{fr:6.1f}{tag}  "

        print(row)

    print("="*55 + "\n")

def diagnose_nest():
    nest.ResetKernel()
    n = nest.Create("iaf_cond_exp", 1)
    mm = nest.Create("multimeter", params={"record_from": ["g_ex", "g_in"]})
    nest.Connect(mm, n)
    pg = nest.Create("poisson_generator", params={"rate": 1000.0})
    nest.Connect(pg, n, syn_spec={"weight": -2.0, "delay": 1.5})
    nest.Simulate(500.0)
    ev = mm.get("events")
    print("\n=== NEST DIAGNOSTIC ===")
    print(f"g_ex mean: {ev['g_ex'].mean():.6f}")
    print(f"g_in mean: {ev['g_in'].mean():.6f}")
    print("=======================\n")
 
# ============================================================
# Main execution
# ============================================================
if __name__ == "__main__":

    N_TOTAL   = 5000
    N_RUNS    = 5
    SEED_BASE = 42

    conditions = [
        ("AD",  2.5),
      #  ("MCI", 3.5),
        ("HC",  6.5),
    ]

    #diagnose_nest() 
    #find_working_params()

    print("=" * 60)
    print(f"E/I EEG proxy — {N_RUNS} runs per condition, N={N_TOTAL}")
    print("=" * 60)

    # Run each condition N_RUNS times, collect spectra
    collected = {name: {"Pxx": [], "rates": [], "f": None} for name, _ in conditions}

    for name, g_ratio in conditions:
        for run_idx in range(N_RUNS):
            seed = SEED_BASE + run_idx * 1000 + int(10 * g_ratio)
            print(f"\n--- {name} run {run_idx+1}/{N_RUNS} ---")
            res = run_simulation(name, g_ratio, N_total=N_TOTAL, seed=seed)
            collected[name]["Pxx"].append(res["Pxx"])
            collected[name]["rates"].append(res["firing_rate"])
            collected[name]["f"] = res["f"]

    # Average across runs
    results = []
    for name, _ in conditions:
        pxx_arr  = np.array(collected[name]["Pxx"])
        mean_pxx = pxx_arr.mean(axis=0)
        std_pxx  = pxx_arr.std(axis=0)
        mean_fr  = np.mean(collected[name]["rates"])
        std_fr   = np.std(collected[name]["rates"])
        print(f"\n{name}: FR = {mean_fr:.2f} ± {std_fr:.2f} Hz")
        results.append({"condition": name, "f": collected[name]["f"],
                        "Pxx_mean": mean_pxx, "Pxx_std": std_pxx})

    colors = {"AD": "#90EE90", "MCI": "#FFD700", "HC": "#A9A9A9"}

    def make_plot(show_sd):
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        for r in results:
            col = colors.get(r["condition"], "gray")
            ax.plot(r["f"], r["Pxx_mean"], label=r["condition"],
                    linewidth=2.5, color=col)
            if show_sd:
                ax.fill_between(r["f"],
                                r["Pxx_mean"] - r["Pxx_std"],
                                r["Pxx_mean"] + r["Pxx_std"],
                                color=col, alpha=0.2)

        for boundary in [4, 8, 13, 30]:
            ax.axvline(x=boundary, color='gray', linestyle='--',
                       linewidth=1.5, alpha=0.6)

        y_max = ax.get_ylim()[1]
        for center, bname in zip([(1+4)/2, (4+8)/2, (8+13)/2, (13+30)/2, (30+40)/2],
                                  ['Delta', 'Theta', 'Alpha', 'Beta', 'Gamma']):
            if center <= 40:
                ax.text(center, y_max * 0.95, bname, ha='center',
                        fontsize=10, style='italic', color='gray', alpha=0.7)

        ax.set_xlabel("Frequency (Hz)", fontsize=14, fontweight='bold')
        ax.set_ylabel("Relative power",  fontsize=14, fontweight='bold')
        ax.set_xlim(1, 40)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=11, loc='upper right')
        plt.tight_layout()
        return fig

    # Save both versions
    fig = make_plot(show_sd=False)
    fig.savefig(f"EI_EEG_proxy_mean_only_N{N_TOTAL}.png", dpi=300)
    fig.savefig(f"EI_EEG_proxy_mean_only_N{N_TOTAL}.pdf")
    plt.close(fig)
    print(f"\n✓ Saved: EI_EEG_proxy_mean_only_N{N_TOTAL}.png / .pdf")

    fig = make_plot(show_sd=True)
    fig.savefig(f"EI_EEG_proxy_mean_sd_N{N_TOTAL}.png", dpi=300)
    fig.savefig(f"EI_EEG_proxy_mean_sd_N{N_TOTAL}.pdf")
    plt.close(fig)
    print(f"✓ Saved: EI_EEG_proxy_mean_sd_N{N_TOTAL}.png / .pdf")

    print("\nDone.")
