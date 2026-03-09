# ============================================================
# PyNEST – E/I imbalance model (Conductance-based LIF)
# EEG/MEG proxy from synaptic currents
# Multi-run averaged version
# ============================================================

import nest
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.signal import welch, savgol_filter
from scipy.interpolate import interp1d


def run_simulation(condition_name,
                   g_ratio,
                   N_total=400,
                   frac_exc=0.8,
                   p_conn=0.2,
                   nu_ext=3.0,
                   sim_time=6000.0,
                   warmup=2000.0,
                   seed=42,
                   record_fraction=0.15,
                   smooth_spectrum=True):
    """
    Run a single E/I network simulation with conductance-based neurons
    and synaptic current EEG/LFP proxy.

    Parameters
    ----------
    condition_name   : str   - label (e.g. 'AD', 'MCI', 'HC')
    g_ratio          : float - g_I / g_E ratio
    N_total          : int   - total number of neurons
    frac_exc         : float - fraction of excitatory neurons
    p_conn           : float - connection probability
    nu_ext           : float - external Poisson rate (kHz)
    sim_time         : float - simulation time after warmup (ms)
    warmup           : float - warmup period (ms)
    seed             : int   - RNG seed
    record_fraction  : float - fraction of E neurons to record from
    smooth_spectrum  : bool  - apply Savitzky-Golay smoothing
    """

    # --------------------
    # NEST kernel setup
    # --------------------
    nest.ResetKernel()
    nest.resolution = 0.1
    nest.set_verbosity("M_WARNING")

    np.random.seed(seed)
    nest.SetKernelStatus({"rng_seed": seed})

    # --------------------
    # Populations
    # --------------------
    N_E = int(frac_exc * N_total)
    N_I = N_total - N_E

    E = nest.Create("iaf_cond_exp", N_E)
    I = nest.Create("iaf_cond_exp", N_I)

    neuron_params = {
        "C_m":        250.0,
        "g_L":        16.67,
        "E_L":        -70.0,
        "V_th":       -50.0,
        "V_reset":    -70.0,
        "t_ref":       2.0,
        "E_ex":        0.0,
        "E_in":       -80.0,
        "tau_syn_ex":  2.0,
        "tau_syn_in":  8.0,
    }

    for pop in (E, I):
        pop.set(neuron_params)
        pop.V_m = -70.0 + 5.0 * np.random.randn(len(pop))

    # --------------------
    # Synaptic strengths
    # --------------------
    g_E = 2.0
    g_I = g_ratio * g_E
    delay = 1.5

    # --------------------
    # External drive
    # --------------------
    ext = nest.Create("poisson_generator")
    ext.rate = nu_ext * 1000.0

    nest.Connect(ext, E, syn_spec={"weight": g_E, "delay": delay})
    nest.Connect(ext, I, syn_spec={"weight": g_E, "delay": delay})

    # --------------------
    # Recurrent connections
    # --------------------
    conn = {"rule": "pairwise_bernoulli", "p": p_conn}

    nest.Connect(E, E, conn, syn_spec={"weight":  g_E,  "delay": delay})
    nest.Connect(E, I, conn, syn_spec={"weight":  g_E,  "delay": delay})
    nest.Connect(I, E, conn, syn_spec={"weight": -g_I,  "delay": delay})
    nest.Connect(I, I, conn, syn_spec={"weight": -g_I,  "delay": delay})

    # --------------------
    # Multimeter
    # --------------------
    n_rec = max(50, int(record_fraction * N_E))
    n_rec = min(n_rec, N_E)

    mm = nest.Create("multimeter")
    mm.set({"interval": 1.0, "record_from": ["g_ex", "g_in", "V_m"]})
    nest.Connect(mm, E[:n_rec])

    # --------------------
    # Spike recorder
    # --------------------
    spike_rec = nest.Create("spike_recorder")
    nest.Connect(E, spike_rec)

    # --------------------
    # Simulate
    # --------------------
    nest.Simulate(warmup)
    nest.Simulate(sim_time)

    # --------------------
    # EEG / LFP proxy from synaptic currents
    # --------------------
    ev      = mm.get("events")
    times   = np.array(ev["times"])
    senders = np.array(ev["senders"])
    g_ex    = np.array(ev["g_ex"])
    g_in    = np.array(ev["g_in"])
    V_m     = np.array(ev["V_m"])

    E_ex_rev = 0.0
    E_in_rev = -80.0

    I_ex = g_ex * (V_m - E_ex_rev)
    I_in = g_in * (V_m - E_in_rev)

    mask            = times > warmup
    times_f         = times[mask]
    senders_f       = senders[mask]
    I_ex_f          = I_ex[mask]
    I_in_f          = I_in[mask]
    V_m_f           = V_m[mask]

    unique_times   = np.sort(np.unique(times_f))
    unique_neurons = np.sort(np.unique(senders_f))
    n_times        = len(unique_times)
    n_neurons      = len(unique_neurons)

    if n_times < 1000:
        return {'success': False}

    expected = n_times * n_neurons
    if len(times_f) != expected:
        I_ex_mat = np.zeros((n_times, n_neurons))
        I_in_mat = np.zeros((n_times, n_neurons))
        n2i = {nid: idx for idx, nid in enumerate(unique_neurons)}
        t2i = {t:   idx for idx, t   in enumerate(unique_times)}
        for k in range(len(times_f)):
            ti = t2i[times_f[k]]
            ni = n2i[senders_f[k]]
            I_ex_mat[ti, ni] = I_ex_f[k]
            I_in_mat[ti, ni] = I_in_f[k]
    else:
        I_ex_mat = I_ex_f.reshape(n_times, n_neurons)
        I_in_mat = I_in_f.reshape(n_times, n_neurons)

    lfp  = I_ex_mat.mean(axis=1) - I_in_mat.mean(axis=1)
    lfp -= lfp.mean()

    # --------------------
    # Spike rate
    # --------------------
    spike_times = spike_rec.get("events")["times"]
    spike_times = spike_times[spike_times > warmup]
    firing_rate = len(spike_times) / (sim_time / 1000.0) / N_E

    # --------------------
    # Power spectrum
    # --------------------
    fs      = 1000.0
    nperseg = min(len(lfp) // 2, 16384)
    nperseg = max(nperseg, 4096)
    noverlap = int(15 * nperseg // 16)

    f, Pxx = welch(lfp, fs=fs, nperseg=nperseg, noverlap=noverlap,
                   window='hann', detrend='constant')

    band = (f >= 1) & (f <= 40)
    f    = f[band]
    Pxx  = Pxx[band]

    if smooth_spectrum:
        wl = min(21, len(Pxx) // 2 * 2 - 1)
        if wl >= 5:
            Pxx = savgol_filter(Pxx, window_length=wl, polyorder=3)

    Pxx = np.maximum(Pxx, 0)
    Pxx /= Pxx.sum()

    return {
        'condition':    condition_name,
        'g_ratio':      g_ratio,
        'f':            f,
        'Pxx':          Pxx,
        'firing_rate':  firing_rate,
        'n_rec':        n_rec,
        'N_E':          N_E,
        'success':      True
    }


def run_multiple_simulations(condition_name,
                              g_ratio,
                              n_runs=10,
                              seed_base=42,
                              **sim_kwargs):
    """
    Run multiple simulations for one condition and return averaged spectra.
    """
    print(f"\n[{condition_name}] g_I/g_E = {g_ratio}")
    print(f"    Running {n_runs} simulations...")

    all_f      = []
    all_Pxx    = []
    all_rates  = []

    for run_idx in range(n_runs):
        seed = seed_base + run_idx * 1000 + int(10 * g_ratio)
        print(f"    Run {run_idx + 1}/{n_runs} (seed={seed})...", end=" ", flush=True)

        res = run_simulation(condition_name, g_ratio, seed=seed, **sim_kwargs)

        if res.get('success', False):
            all_f.append(res['f'])
            all_Pxx.append(res['Pxx'])
            all_rates.append(res['firing_rate'])
            print(f"✓ (FR={res['firing_rate']:.2f} Hz)")
        else:
            print("✗ Failed")

    if len(all_Pxx) == 0:
        print(f"    All simulations failed for {condition_name}")
        return None

    # Interpolate all spectra to common frequency grid
    f_common = all_f[0]
    powers_interp = []

    for f_i, pxx_i in zip(all_f, all_Pxx):
        if len(f_i) == len(f_common) and np.allclose(f_i, f_common):
            powers_interp.append(pxx_i)
        else:
            fn = interp1d(f_i, pxx_i, kind='linear',
                          bounds_error=False, fill_value='extrapolate')
            powers_interp.append(fn(f_common))

    powers_arr = np.array(powers_interp)

    mean_power      = np.mean(powers_arr, axis=0)
    std_power       = np.std(powers_arr,  axis=0)
    mean_rate       = np.mean(all_rates)
    std_rate        = np.std(all_rates)

    print(f"    ✓ Avg firing rate : {mean_rate:.2f} ± {std_rate:.2f} Hz")
    print(f"    ✓ Successful runs : {len(all_Pxx)}/{n_runs}")

    return {
        'condition':        condition_name,
        'g_ratio':          g_ratio,
        'f':                f_common,
        'Pxx_mean':         mean_power,
        'Pxx_std':          std_power,
        'firing_rate_mean': mean_rate,
        'firing_rate_std':  std_rate,
        'n_successful':     len(all_Pxx),
        'success':          True
    }


# ============================================================
# Main execution
# ============================================================
if __name__ == "__main__":

    N_TOTAL          = 400
    N_RUNS           = 10
    SEED_BASE        = 42
    RECORD_FRACTION  = 0.15
    SMOOTH_SPECTRUM  = True

    conditions = [
        ("AD",  2.5),
        ("MCI", 3.5),
        ("HC",  6.5),
    ]

    sim_kwargs = dict(
        N_total         = N_TOTAL,
        frac_exc        = 0.8,
        p_conn          = 0.2,
        nu_ext          = 3.0,
        sim_time        = 6000.0,
        warmup          = 2000.0,
        record_fraction = RECORD_FRACTION,
        smooth_spectrum = SMOOTH_SPECTRUM,
    )

    print("=" * 60)
    print("E/I Balance – EEG proxy, multi-run averaged")
    print(f"Network: {N_TOTAL} neurons  |  {N_RUNS} runs/condition")
    print("=" * 60)

    all_spectra = []
    for name, g_ratio in conditions:
        res = run_multiple_simulations(
            name, g_ratio,
            n_runs    = N_RUNS,
            seed_base = SEED_BASE,
            **sim_kwargs
        )
        if res and res['success']:
            all_spectra.append(res)

    print("\n" + "=" * 60)
    print(f"Completed: {len(all_spectra)}/{len(conditions)} conditions")
    print("=" * 60)
    for r in all_spectra:
        print(f"  {r['condition']:3s}: FR = {r['firing_rate_mean']:.2f} ± "
              f"{r['firing_rate_std']:.2f} Hz  "
              f"({r['n_successful']}/{N_RUNS} runs)")

    # --------------------
    # Plot
    # --------------------
    if all_spectra:
        fig, ax = plt.subplots(figsize=(10, 6))

        colors = {
            "AD":  "#90EE90",
            "MCI": "#FFD700",
            "HC":  "#A9A9A9",
        }

        for s in all_spectra:
            col = colors.get(s['condition'], 'gray')

            # Mean line
            ax.plot(s['f'], s['Pxx_mean'],
                    label=s['condition'],
                    linewidth=2.5,
                    color=col,
                    alpha=0.85)

            # Shaded ±1 SD
            ax.fill_between(
                s['f'],
                np.maximum(s['Pxx_mean'] - s['Pxx_std'], 0),
                s['Pxx_mean'] + s['Pxx_std'],
                color=col,
                alpha=0.2
            )

        # Frequency band boundaries
        for boundary in [4, 8, 13, 30]:
            ax.axvline(x=boundary, color='gray', linestyle='--',
                       linewidth=1.5, alpha=0.6)

        # Band labels
        y_max = ax.get_ylim()[1]
        band_centers = [(1+4)/2, (4+8)/2, (8+13)/2, (13+30)/2, (30+40)/2]
        band_names   = ['Delta', 'Theta', 'Alpha', 'Beta', 'Gamma']
        for center, name in zip(band_centers, band_names):
            if center <= 40:
                ax.text(center, y_max * 0.95, name,
                        ha='center', fontsize=10,
                        style='italic', color='gray', alpha=0.7)

        ax.set_xlabel("Frequency (Hz)", fontsize=14, fontweight='bold')
        ax.set_ylabel("Relative power",  fontsize=14, fontweight='bold')
        ax.set_xlim(1, 40)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=11, loc='upper right')

        plt.tight_layout()

        out_png = f"EI_EEG_proxy_multirun_N{N_TOTAL}.png"
        out_pdf = f"EI_EEG_proxy_multirun_N{N_TOTAL}.pdf"

        plt.savefig(out_png, dpi=300)
        plt.savefig(out_pdf)

        print(f"\n✓ Plots saved:")
        print(f"  - {out_png}")
        print(f"  - {out_pdf}")

    print("\nDone.")
