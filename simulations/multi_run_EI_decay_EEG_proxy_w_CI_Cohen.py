# ============================================================
# PyNEST prototype
# E/I imbalance model
# g_ratio = g_I / g_E  (higher -> stronger inhibition)
# Modified to run multiple simulations and average results
# + bootstrapped 95 % CI per frequency bin
# + Cohen's d (pooled SD) spectrum AD vs HC
# ============================================================

import nest
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.signal import welch
from scipy.interpolate import interp1d


# ============================================================
# Bootstrap CI helper
# ============================================================

def bootstrap_ci(powers_array, n_boot=2000, alpha=0.05, rng_seed=0):
    """
    Compute per-bin bootstrapped confidence interval.

    Parameters
    ----------
    powers_array : ndarray, shape (n_runs, n_freqs)
        One row per simulation run.
    n_boot : int
        Number of bootstrap resamples.
    alpha : float
        Two-tailed significance level (default 0.05 → 95 % CI).
    rng_seed : int
        Random seed for reproducibility.

    Returns
    -------
    ci_low, ci_high : ndarray, shape (n_freqs,)
        Lower and upper CI bounds.
    """
    rng = np.random.default_rng(rng_seed)
    n_runs, n_freqs = powers_array.shape
    boot_means = np.empty((n_boot, n_freqs))
    for b in range(n_boot):
        idx = rng.integers(0, n_runs, size=n_runs)
        boot_means[b] = powers_array[idx].mean(axis=0)
    ci_low  = np.percentile(boot_means, 100 * alpha / 2,     axis=0)
    ci_high = np.percentile(boot_means, 100 * (1 - alpha / 2), axis=0)
    return ci_low, ci_high


# ============================================================
# Cohen's d (pooled SD) helper
# ============================================================

def cohens_d_pooled(mu1, sd1, n1, mu2, sd2, n2):
    """
    Per-bin Cohen's d using the pooled standard deviation.

        d[k] = (mu1[k] - mu2[k]) / s_pooled[k]

        s_pooled = sqrt( ((n1-1)*sd1^2 + (n2-1)*sd2^2) / (n1+n2-2) )

    Parameters
    ----------
    mu1, sd1, n1 : array-like, array-like, int
        Mean, SD and sample size for group 1.
    mu2, sd2, n2 : array-like, array-like, int
        Mean, SD and sample size for group 2.

    Returns
    -------
    d : ndarray
        Cohen's d per frequency bin (positive = group 1 > group 2).
    s_pooled : ndarray
        Pooled SD per bin (useful for sanity checks).
    """
    mu1, sd1 = np.asarray(mu1), np.asarray(sd1)
    mu2, sd2 = np.asarray(mu2), np.asarray(sd2)
    s_pooled = np.sqrt(((n1 - 1) * sd1**2 + (n2 - 1) * sd2**2) / (n1 + n2 - 2))
    # Avoid division by zero in degenerate bins
    with np.errstate(invalid='ignore', divide='ignore'):
        d = np.where(s_pooled > 0, (mu1 - mu2) / s_pooled, 0.0)
    return d, s_pooled


# ============================================================
# Single simulation
# ============================================================

def run_simulation(condition_name,
                   g_ratio,
                   scale_flag,
                   N_total=5000,
                   frac_exc=0.8,
                   p_conn=0.2,
                   nu_ext=5.0,
                   sim_time=5000.0,
                   warmup=1000.0,
                   seed_base=42):
    """Run an E/I balanced network simulation with robust analysis."""

    # --------------------
    # Kernel setup
    # --------------------
    nest.ResetKernel()
    nest.set_verbosity("M_WARNING")
    nest.resolution = 0.1  # ms

    rng_seed = seed_base
    np.random.seed(rng_seed)

    # --------------------
    # Populations
    # --------------------
    N_E = int(N_total * frac_exc)
    N_I = N_total - N_E

    if scale_flag is True:
        K_E_target = 30
        p_conn = K_E_target / (N_total * frac_exc)

    E_pop = nest.Create("iaf_cond_exp", N_E)
    I_pop = nest.Create("iaf_cond_exp", N_I)

    # --------------------
    # Neuronal heterogeneity
    # --------------------
    E_pop.V_m  = -70.0 + 5.0 * np.random.randn(N_E)
    I_pop.V_m  = -70.0 + 5.0 * np.random.randn(N_I)

    E_pop.V_th = -50.0 + 2.0 * np.random.randn(N_E)
    I_pop.V_th = -50.0 + 2.0 * np.random.randn(N_I)

    E_pop.t_ref = 2.0
    I_pop.t_ref = 1.0

    E_pop.tau_syn_ex = 2.0
    E_pop.tau_syn_in = 4.0
    I_pop.tau_syn_ex = 1.0
    I_pop.tau_syn_in = 2.0

    # --------------------
    # External Poisson drive
    # --------------------
    g_E_base = 3
    delay    = 1.5

    ext_drives = []
    for _ in range(N_total):
        pg   = nest.Create("poisson_generator")
        rate = nu_ext * 1000.0 * np.random.lognormal(mean=0.0, sigma=0.2)
        pg.rate = rate
        ext_drives.append(pg)

    for i, pg in enumerate(ext_drives):
        if i < N_E:
            nest.Connect(pg, E_pop[i:i+1],
                         syn_spec={"weight": g_E_base, "delay": delay})
        else:
            nest.Connect(pg, I_pop[i-N_E:i-N_E+1],
                         syn_spec={"weight": g_E_base, "delay": delay})

    noise      = nest.Create("poisson_generator")
    noise.rate = 200.0
    nest.Connect(noise, E_pop, syn_spec={"weight": 0.3 * g_E_base, "delay": delay})
    nest.Connect(noise, I_pop, syn_spec={"weight": 0.3 * g_E_base, "delay": delay})

    # --------------------
    # Recurrent connectivity
    # --------------------
    g_E       = g_E_base
    g_I       = g_ratio * g_E
    conn_spec = {"rule": "pairwise_bernoulli", "p": p_conn}

    nest.Connect(E_pop, E_pop, conn_spec, syn_spec={"weight":  g_E,        "delay": delay})
    nest.Connect(E_pop, I_pop, conn_spec, syn_spec={"weight":  1.2 * g_E,  "delay": delay})
    nest.Connect(I_pop, E_pop, conn_spec, syn_spec={"weight": -g_I,        "delay": delay})
    nest.Connect(I_pop, I_pop, conn_spec, syn_spec={"weight": -0.8 * g_I,  "delay": delay})

    # --------------------
    # Recording
    # --------------------
    mE = nest.Create("multimeter")
    mE.set(record_from=["V_m"], interval=1.0)
    nest.Connect(mE, E_pop[:20])

    spike_rec = nest.Create("spike_recorder")
    nest.Connect(E_pop, spike_rec)

    # --------------------
    # Simulation
    # --------------------
    nest.Simulate(warmup)
    nest.Simulate(sim_time)

    # --------------------
    # Spike-rate analysis
    # --------------------
    spikes      = spike_rec.get("events")
    spike_times = spikes["times"]
    spike_times = spike_times[spike_times > warmup]
    spike_rate  = len(spike_times) / (N_E * sim_time / 1000.0)

    # --------------------
    # LFP proxy via Welch
    # --------------------
    events = mE.get("events")

    if "times" not in events or len(events["times"]) < 2000:
        return {'success': False}

    t   = np.array(events["times"])
    V_m = np.array(events["V_m"])

    mask = t > warmup
    V_m  = V_m[mask]

    if len(V_m) < 1000:
        return {'success': False}

    fs      = 1000.0 / mE.interval
    nperseg = min(4096, len(V_m) // 4)

    f, Pxx = welch(V_m, fs=fs, nperseg=nperseg, noverlap=3 * nperseg // 4)

    band = (f >= 1) & (f <= 40)
    f    = f[band]
    Pxx  = Pxx[band]
    Pxx_rel = Pxx / np.sum(Pxx)

    return {
        'condition':  condition_name,
        'g_ratio':    g_ratio,
        'f':          f,
        'Pxx':        Pxx_rel,
        'spike_rate': spike_rate,
        'success':    True
    }


# ============================================================
# Multiple simulations → average + CI
# ============================================================

def run_multiple_simulations(condition_name, g_ratio, n_runs=10, seed_base=42):
    """Run multiple simulations; return mean, SD, bootstrapped 95 % CI."""

    print(f"\n[{condition_name}] g_I/g_E = {g_ratio}")
    print(f"    Running {n_runs} simulations...")

    all_frequencies = []
    all_powers      = []
    spike_rates     = []

    for run_idx in range(n_runs):
        seed = seed_base + run_idx * 1000 + int(10 * g_ratio)
        print(f"    Run {run_idx + 1}/{n_runs} (seed={seed})...", end=" ")

        res = run_simulation(condition_name, g_ratio, scale_flag, seed_base=seed)

        if res.get('success', False):
            all_frequencies.append(res['f'])
            all_powers.append(res['Pxx'])
            spike_rates.append(res['spike_rate'])
            print(f"✓ (rate={res['spike_rate']:.2f} Hz)")
        else:
            print("✗ Failed")

    if len(all_powers) == 0:
        print(f"    All simulations failed for {condition_name}")
        return None

    # Interpolate to a common frequency grid
    f_common           = all_frequencies[0]
    powers_interpolated = []

    for f, pxx in zip(all_frequencies, all_powers):
        if len(f) == len(f_common) and np.allclose(f, f_common):
            powers_interpolated.append(pxx)
        else:
            interp_func = interp1d(f, pxx, kind='linear',
                                   bounds_error=False, fill_value='extrapolate')
            powers_interpolated.append(interp_func(f_common))

    powers_array = np.array(powers_interpolated)   # shape (n_runs, n_freqs)

    mean_power       = np.mean(powers_array, axis=0)
    std_power        = np.std( powers_array, axis=0, ddof=1)
    mean_spike_rate  = np.mean(spike_rates)
    std_spike_rate   = np.std( spike_rates)

    # ----------------------------------------------------------
    # Bootstrapped 95 % CI per frequency bin
    # ----------------------------------------------------------
    ci_low, ci_high = bootstrap_ci(powers_array, n_boot=2000, alpha=0.05,
                                   rng_seed=int(10 * g_ratio))

    print(f"    ✓ Average spike rate : {mean_spike_rate:.2f} ± {std_spike_rate:.2f} Hz")
    print(f"    ✓ Successful runs     : {len(all_powers)}/{n_runs}")
    print(f"    ✓ 95 % CI computed (bootstrap, n_boot=2000)")

    return {
        'condition':        condition_name,
        'g_ratio':          g_ratio,
        'f':                f_common,
        'Pxx_mean':         mean_power,
        'Pxx_std':          std_power,
        'ci_low':           ci_low,
        'ci_high':          ci_high,
        'n_successful':     len(all_powers),
        'spike_rate_mean':  mean_spike_rate,
        'spike_rate_std':   std_spike_rate,
        'success':          True
    }


# ============================================================
# Main execution
# ============================================================

conditions = [
    ("AD", 2.5),   # Low inhibition
    ("HC", 6.5),   # High inhibition
]

all_spectra          = []
n_runs_per_condition = 10
scale_flag           = False

print("Starting simulations")
print("=" * 60)
print(f"Running {n_runs_per_condition} simulations per condition")

for name, g_ratio in conditions:
    res = run_multiple_simulations(name, g_ratio, n_runs=n_runs_per_condition)
    if res and res.get('success', False):
        all_spectra.append(res)

print("\n" + "=" * 60)
print(f"Completed conditions: {len(all_spectra)}/{len(conditions)}")

# ============================================================
# Cohen's d (pooled SD) — AD vs HC
# ============================================================

spectra_by_name = {s['condition']: s for s in all_spectra}

if 'AD' in spectra_by_name and 'HC' in spectra_by_name:
    ad = spectra_by_name['AD']
    hc = spectra_by_name['HC']

    d_spectrum, s_pooled = cohens_d_pooled(
        mu1=ad['Pxx_mean'], sd1=ad['Pxx_std'], n1=ad['n_successful'],
        mu2=hc['Pxx_mean'], sd2=hc['Pxx_std'], n2=hc['n_successful']
    )

    f_ref   = ad['f']    # both conditions share the same grid after interpolation
    abs_d   = np.abs(d_spectrum)
    peak_idx = np.argmax(abs_d)

    print("\nCohen's d (pooled SD) — AD vs HC")
    print("-" * 40)
    print(f"  Mean |d| across 1–40 Hz : {abs_d.mean():.3f}")
    print(f"  Peak |d|                : {abs_d[peak_idx]:.3f}  at {f_ref[peak_idx]:.1f} Hz")
    print(f"  Sign convention         : positive d = AD > HC")

    # Per-band summary
    bands = [("Delta", 1, 4), ("Theta", 4, 8),
             ("Alpha", 8, 13), ("Beta", 13, 30), ("Gamma", 30, 40)]
    print("\n  Per-band mean |d|:")
    for bname, flo, fhi in bands:
        mask = (f_ref >= flo) & (f_ref < fhi)
        if mask.any():
            print(f"    {bname:6s} ({flo:2d}–{fhi:2d} Hz): {abs_d[mask].mean():.3f}")
else:
    d_spectrum = None
    f_ref      = None
    print("\n  Skipping Cohen's d (need both AD and HC results)")

# ============================================================
# Plotting
# ============================================================

colors = {
    "AD":  "#90EE90",
    "MCI": "#FFD700",
    "HC":  "#A9A9A9",
}

if all_spectra:

    # -----------------------------------------------------------
    # Figure layout: 2 rows
    #   top  — mean ± 95 % CI power spectra
    #   bottom — Cohen's d spectrum (only when both groups present)
    # -----------------------------------------------------------
    n_rows = 2 if d_spectrum is not None else 1
    fig, axes = plt.subplots(n_rows, 1,
                             figsize=(10, 5 * n_rows),
                             sharex=True)
    if n_rows == 1:
        axes = [axes]

    ax = axes[0]

    for s in all_spectra:
        ax.plot(s['f'], s['Pxx_mean'],
                label=s['condition'],
                linewidth=2.5,
                color=colors[s['condition']],
                alpha=0.85)

        # 95 % bootstrapped CI (replaces plain SD shade)
        ax.fill_between(s['f'],
                        s['ci_low'],
                        s['ci_high'],
                        color=colors[s['condition']],
                        alpha=0.25,
                        label=f"{s['condition']} 95 % CI")

    band_boundaries = [4, 8, 13, 30]
    for bnd in band_boundaries:
        ax.axvline(x=bnd, color='gray', linestyle='--', linewidth=1.5, alpha=0.6)

    y_max = ax.get_ylim()[1]
    band_centers = [(1+4)/2, (4+8)/2, (8+13)/2, (13+30)/2, (30+40)/2]
    band_names   = ['Delta', 'Theta', 'Alpha', 'Beta', 'Gamma']
    for center, bname in zip(band_centers, band_names):
        if center <= 40:
            ax.text(center, y_max * 0.95, bname,
                    ha='center', fontsize=10, style='italic',
                    color='gray', alpha=0.7)

    ax.set_ylabel("Relative power", fontsize=14, fontweight='bold')
    ax.set_xlim([1, 40])
    ax.grid(alpha=0.3)
    ax.legend(fontsize=10, loc='upper right', ncol=2)
    ax.set_title("Power spectra (mean ± 95 % bootstrap CI)", fontsize=13)

    # -----------------------------------------------------------
    # Bottom panel: Cohen's d spectrum
    # -----------------------------------------------------------
    if d_spectrum is not None:
        ax2 = axes[1]

        # Fill by sign for readability
        ax2.fill_between(f_ref, d_spectrum, 0,
                         where=(d_spectrum >= 0),
                         color="#8A6BB5", alpha=0.4, label="AD > HC")
        ax2.fill_between(f_ref, d_spectrum, 0,
                         where=(d_spectrum < 0),
                         color="#4A9B7F", alpha=0.4, label="HC > AD")

        ax2.plot(f_ref, d_spectrum, color="#5C4080", linewidth=2.0, alpha=0.9)

        # Reference effect-size lines
        for level, ls, lbl in [(0.2, ':', 'small'), (0.5, '--', 'medium'), (0.8, '-.', 'large')]:
            ax2.axhline( level, color='#888', linestyle=ls, linewidth=1.2, alpha=0.7)
            ax2.axhline(-level, color='#888', linestyle=ls, linewidth=1.2, alpha=0.7)
            ax2.text(40.2, level,  lbl, va='center', fontsize=9, color='#888')
            ax2.text(40.2, -level, lbl, va='center', fontsize=9, color='#888')

        ax2.axhline(0, color='gray', linewidth=1.0, alpha=0.5)

        # Band boundaries (shared x-axis, so lines inherited, but add labels)
        for bnd in band_boundaries:
            ax2.axvline(x=bnd, color='gray', linestyle='--', linewidth=1.5, alpha=0.6)

        ax2.set_xlabel("Frequency (Hz)", fontsize=14, fontweight='bold')
        ax2.set_ylabel("Cohen's d (AD − HC)", fontsize=14, fontweight='bold')
        ax2.set_xlim([1, 40])
        ax2.grid(alpha=0.3)
        ax2.legend(fontsize=10, loc='upper right')
        ax2.set_title(
            f"Cohen's d pooled (mean |d| = {np.abs(d_spectrum).mean():.3f}, "
            f"peak |d| = {np.abs(d_spectrum).max():.3f} @ "
            f"{f_ref[np.argmax(np.abs(d_spectrum))]:.1f} Hz)",
            fontsize=12
        )

    plt.tight_layout()
    plt.savefig("power_spectra_EI_averaged.png", dpi=300)
    plt.savefig("power_spectra_EI_averaged.pdf")

    print("\n✓ Plots saved:")
    print("  - power_spectra_EI_averaged.png")
    print("  - power_spectra_EI_averaged.pdf")

    # Separate standalone Cohen's d figure
    if d_spectrum is not None:
        fig2, ax3 = plt.subplots(figsize=(10, 4))

        ax3.fill_between(f_ref, d_spectrum, 0,
                         where=(d_spectrum >= 0),
                         color="#8A6BB5", alpha=0.4, label="AD > HC")
        ax3.fill_between(f_ref, d_spectrum, 0,
                         where=(d_spectrum < 0),
                         color="#4A9B7F", alpha=0.4, label="HC > AD")
        ax3.plot(f_ref, d_spectrum, color="#5C4080", linewidth=2.0)

        for level, ls, lbl in [(0.2, ':', 'small'), (0.5, '--', 'medium'), (0.8, '-.', 'large')]:
            ax3.axhline( level, color='#888', linestyle=ls, linewidth=1.2, alpha=0.7)
            ax3.axhline(-level, color='#888', linestyle=ls, linewidth=1.2, alpha=0.7)
            ax3.text(40.3, level,  lbl, va='center', fontsize=9, color='#888')
            ax3.text(40.3, -level, lbl, va='center', fontsize=9, color='#888')

        ax3.axhline(0, color='gray', linewidth=1.0, alpha=0.5)

        for bnd in band_boundaries:
            ax3.axvline(x=bnd, color='gray', linestyle='--', linewidth=1.5, alpha=0.6)

        y_max3 = max(np.abs(d_spectrum)) * 1.1
        for center, bname in zip(band_centers, band_names):
            if center <= 40:
                ax3.text(center, y_max3 * 0.92, bname,
                         ha='center', fontsize=10, style='italic',
                         color='gray', alpha=0.7)

        ax3.set_xlabel("Frequency (Hz)", fontsize=14, fontweight='bold')
        ax3.set_ylabel("Cohen's d  (AD − HC, pooled SD)", fontsize=14, fontweight='bold')
        ax3.set_xlim([1, 40])
        ax3.legend(fontsize=11, loc='upper right')
        ax3.grid(alpha=0.3)
        ax3.set_title(
            "Cohen's d spectrum  |  AD vs HC  "
            f"(n_AD={ad['n_successful']}, n_HC={hc['n_successful']})",
            fontsize=13
        )

        plt.tight_layout()
        plt.savefig("cohens_d_EI.png", dpi=300)
        plt.savefig("cohens_d_EI.pdf")
        print("  - cohens_d_EI.png")
        print("  - cohens_d_EI.pdf")

print("\nDone.")
