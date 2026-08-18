import nest
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.signal import welch, savgol_filter
from scipy.interpolate import interp1d
import os


# ============================================================
# Bootstrap CI helper  (identical to EI-imbalance script)
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
    ci_low  = np.percentile(boot_means, 100 * alpha / 2,       axis=0)
    ci_high = np.percentile(boot_means, 100 * (1 - alpha / 2), axis=0)
    return ci_low, ci_high


# ============================================================
# Log-domain Savitzky-Golay smoothing (plotting only)
# ============================================================
# Power spectra are strictly positive and roughly log-linear/log-power-law
# in shape, so smoothing in log-domain (rather than linear) avoids
# over-smoothing small values relative to large ones and avoids producing
# negative "power" near dips. This is applied ONLY for the plotted
# line/shaded band -- the band table and Cohen's d use the raw (unsmoothed)
# per-bin values, since smoothing before those would blur the actual
# per-bin effect being tested.

def smooth_log_domain(y, window_length=15, polyorder=2):
    y = np.asarray(y, dtype=float)
    n = len(y)

    wl = min(window_length, n)
    if wl % 2 == 0:
        wl -= 1  # savgol_filter requires an odd window length
    if wl < polyorder + 2:
        # Too few points to smooth meaningfully with this polyorder.
        return y

    log_y = np.log(np.clip(y, 1e-300, None))
    smoothed_log = savgol_filter(log_y, window_length=wl, polyorder=polyorder)
    return np.exp(smoothed_log)


# ============================================================
# Cohen's d (pooled SD) helper  (identical to EI-imbalance script)
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
    s_pooled = np.sqrt(
        ((n1 - 1) * sd1**2 + (n2 - 1) * sd2**2) / (n1 + n2 - 2)
    )
    with np.errstate(invalid='ignore', divide='ignore'):
        d = np.where(s_pooled > 0, (mu1 - mu2) / s_pooled, 0.0)
    return d, s_pooled


# ============================================================
# Load connectivity
# ============================================================

def load_connectivity_matrix(group, band, data_root="../ds004504/"):
    conn_file = os.path.join(
        data_root, group,
        f'connectivity_{band}',
        f'average_{band}_plv.npy'
    )
    if not os.path.exists(conn_file):
        raise FileNotFoundError(f"Connectivity file not found: {conn_file}")

    conn_matrix = np.load(conn_file)
    np.fill_diagonal(conn_matrix, 0)
    print(f"    Loaded {band} FC: range [{conn_matrix.min():.3f}, {conn_matrix.max():.3f}]")
    return conn_matrix


# ============================================================
# Subnetwork selection
# ============================================================

def select_subnetworks(conn_matrix, n_subnets=15):
    node_strength    = np.sum(np.abs(conn_matrix), axis=1)
    selected_indices = np.argsort(node_strength)[-n_subnets:]
    subnet_conn      = conn_matrix[np.ix_(selected_indices, selected_indices)]
    return selected_indices, subnet_conn


# ============================================================
# Single simulation
# ============================================================

def run_simulation_with_fc(condition_name,
                           band_name,
                           g_ratio,
                           conn_matrix,
                           N_subnet=21,
                           N_subnets=19,
                           frac_exc=0.8,
                           p_conn_local=0.15,
                           coupling_strength=0.5,
                           nu_ext=0.03,
                           sim_time=5000.0,
                           warmup=1000.0,
                           seed_base=42):
    # NOTE on nu_ext: this was 2.0 (-> 2000 Hz external Poisson drive) under
    # iaf_cond_exp, where "weight" scales a conductance integrated over
    # tau_syn_ex/in, so a very high input rate was needed to reach threshold.
    # Under izhikevich, each incoming spike adds "weight" directly and
    # instantaneously to V_m (no synaptic integration), so 2000 Hz at
    # weight ~1.5 mV would massively over-drive the network and push it into
    # a saturated/synchronized regime dominated by the external noise itself
    # rather than by recurrent E/I dynamics -- which is the most likely
    # explanation for AD and HC power spectra being nearly indistinguishable.
    # nu_ext=0.03 (-> 30 Hz/synapse) is a more physiologically reasonable
    # starting point for a delta-current drive; it should be tuned further
    # once you check the spike-rate diagnostics printed below.

    nest.ResetKernel()
    nest.set_verbosity("M_WARNING")
    nest.resolution = 0.1

    rng_seed = seed_base + hash(condition_name + band_name) % 1000
    np.random.seed(rng_seed)

    N_E_subnet = int(N_subnet * frac_exc)
    N_I_subnet = N_subnet - N_E_subnet

    E_populations = []
    I_populations = []

    for i in range(N_subnets):
        # ---- Izhikevich neurons (replaces iaf_cond_exp LIF) --------------
        # Regular-spiking (RS) parameters for excitatory population,
        # fast-spiking (FS) parameters for inhibitory population — the
        # standard Izhikevich choice for cortical E/I networks.
        E_pop = nest.Create("izhikevich", N_E_subnet)
        I_pop = nest.Create("izhikevich", N_I_subnet)

        E_pop.V_m = -70 + 5 * np.random.randn(N_E_subnet)
        I_pop.V_m = -70 + 5 * np.random.randn(N_I_subnet)

        # RS (regular spiking, excitatory)
        E_pop.a = 0.02
        E_pop.b = 0.2
        E_pop.c = -65.0
        E_pop.d = 8.0

        # FS (fast spiking, inhibitory)
        I_pop.a = 0.1
        I_pop.b = 0.2
        I_pop.c = -65.0
        I_pop.d = 2.0

        # NOTE: V_th, tau_syn_ex, tau_syn_in, t_ref do not exist on the
        # izhikevich model (no conductance-based synapses, fixed spike
        # cutoff at V_th=30 mV by default), so they are intentionally
        # not set here — all other parameters/logic are unchanged.

        E_populations.append(E_pop)
        I_populations.append(I_pop)

    g_E_base    = 3.0
    delay_local = 1.5
    delay_inter = 3.0

    # ── External drive ───────────────────────────────────────────────────────
    for subnet_idx in range(N_subnets):
        E_pop = E_populations[subnet_idx]
        I_pop = I_populations[subnet_idx]

        pg_E = nest.Create("poisson_generator")
        pg_E.rate = nu_ext * 1000
        nest.Connect(pg_E, E_pop,
                     conn_spec={"rule": "all_to_all"},
                     syn_spec={"weight": 0.5 * g_E_base, "delay": delay_local})

        pg_I = nest.Create("poisson_generator")
        pg_I.rate = nu_ext * 1000
        nest.Connect(pg_I, I_pop,
                     conn_spec={"rule": "all_to_all"},
                     syn_spec={"weight": 0.5 * g_E_base, "delay": delay_local})

    # ── Local recurrent connectivity ─────────────────────────────────────────
    g_E = g_E_base
    g_I = g_ratio * g_E

    conn_spec = {"rule": "pairwise_bernoulli", "p": p_conn_local}

    for subnet_idx in range(N_subnets):
        E_pop = E_populations[subnet_idx]
        I_pop = I_populations[subnet_idx]

        nest.Connect(E_pop, E_pop, conn_spec,
                     syn_spec={"weight":  g_E,       "delay": delay_local})
        nest.Connect(E_pop, I_pop, conn_spec,
                     syn_spec={"weight":  1.2 * g_E, "delay": delay_local})
        nest.Connect(I_pop, E_pop, conn_spec,
                     syn_spec={"weight": -g_I,        "delay": delay_local})
        nest.Connect(I_pop, I_pop, conn_spec,
                     syn_spec={"weight": -0.8 * g_I,  "delay": delay_local})

    # ── Inter-subnetwork FC-weighted connections ──────────────────────────────
    conn_normalized = (conn_matrix / np.max(conn_matrix)
                       if np.max(conn_matrix) > 0 else conn_matrix)

    for i in range(N_subnets):
        for j in range(N_subnets):
            if i == j:
                continue
            fc_weight = conn_normalized[i, j]
            if fc_weight > 0.1:
                inter_weight = coupling_strength * g_E * fc_weight
                nest.Connect(
                    E_populations[i], E_populations[j],
                    {"rule": "pairwise_bernoulli", "p": 0.05},
                    syn_spec={"weight": inter_weight, "delay": delay_inter}
                )

    # ── Recording ─────────────────────────────────────────────────────────────
    multimeters = []
    for s in range(N_subnets):
        mE = nest.Create("multimeter")
        mE.set(record_from=["V_m"], interval=1.0)
        nest.Connect(mE, E_populations[s])
        multimeters.append(mE)

    spike_recorders = []
    for subnet_idx in range(N_subnets):
        sr = nest.Create("spike_recorder")
        nest.Connect(E_populations[subnet_idx], sr)
        spike_recorders.append(sr)

    nest.Simulate(warmup + sim_time)

    # ── Spike rates ───────────────────────────────────────────────────────────
    spike_rates = []
    for sr in spike_recorders:
        spikes      = sr.get("events")
        spike_times = np.array(spikes["times"])
        spike_times = spike_times[spike_times > warmup]
        rate = len(spike_times) / (N_E_subnet * sim_time / 1000)
        spike_rates.append(rate)

    # ── PSD: network-wide LFP-proxy (see rationale in comments below) ─────────
    subnet_lfp_traces = []

    for mE in multimeters:
        events = mE.get("events")
        t      = np.array(events["times"])
        Vm_raw = np.array(events["V_m"])
        sender = np.array(events["senders"])

        mask               = t > warmup
        t_m, Vm_m, sndr_m = t[mask], Vm_raw[mask], sender[mask]

        uids = np.unique(sndr_m)
        if len(uids) == 0:
            continue

        # ---- Population-average (LFP-proxy) trace for THIS subnet -------
        # Average raw V_m traces across neurons within the subnet. Do NOT
        # Welch here yet -- that happens once, below, on the network-wide
        # trace (see note there for why).
        neuron_traces = []
        ref_times     = None
        for uid in uids:
            idx  = sndr_m == uid
            tr_t = t_m[idx]
            tr_v = Vm_m[idx]
            order = np.argsort(tr_t)
            tr_t, tr_v = tr_t[order], tr_v[order]
            if ref_times is None:
                ref_times = tr_t
            if len(tr_v) != len(ref_times):
                # Multimeter should sample every connected neuron on the
                # same fixed grid; if a trace is short/misaligned (e.g.
                # dropped sample), interpolate it onto the reference grid
                # instead of silently stacking mismatched lengths.
                tr_v = np.interp(ref_times, tr_t, tr_v)
            neuron_traces.append(tr_v)

        if not neuron_traces or ref_times is None:
            continue

        subnet_lfp = np.mean(neuron_traces, axis=0)
        subnet_lfp_traces.append(subnet_lfp)

    if not subnet_lfp_traces:
        return {"success": False}

    # ---- Network-wide (grand) LFP-proxy trace, THEN ONE Welch call ------
    # IMPORTANT: this averages the subnet-level traces (each already a
    # within-subnet average) into a single network-wide trace, and computes
    # ONE Welch PSD on that. Averaging is linear, so mean-of-subnet-means
    # equals the true grand mean across all E neurons in the network (equal
    # neuron count per subnet here), and unlike averaging per-subnet PSDs,
    # this preserves phase relationships BETWEEN subnets. That matters
    # specifically because inter-subnet coupling here is weighted by the
    # real per-condition (HC/AD) functional-connectivity matrices -- if the
    # conditions differ mainly in cross-subnet synchrony, averaging PSDs
    # across subnets would erase exactly that signal.
    # Subnets may differ by a sample or two due to independent warmup
    # masking; trim to the shortest common length before averaging.
    min_len = min(len(tr) for tr in subnet_lfp_traces)
    network_trace = np.mean([tr[:min_len] for tr in subnet_lfp_traces], axis=0)

    nperseg = min(2048, len(network_trace))
    if nperseg < 16:
        return {"success": False}

    f_last, Pxx_network = welch(network_trace, fs=1000.0, nperseg=nperseg,
                                noverlap=3 * nperseg // 4)

    f_band    = (f_last >= 1) & (f_last <= 100)
    f_out     = f_last[f_band]
    Pxx_grand = Pxx_network[f_band]

    # ---- Guard against degenerate/near-silent PSDs -----------------------
    # If the network is effectively silenced (e.g. strong inhibition +
    # reduced external drive leaves V_m essentially flat), Welch power can
    # be ~0 across the board. Normalizing by np.sum(Pxx_grand) then divides
    # ~0 by ~0, producing NaN that propagates silently into every later
    # mean/std/bootstrap/plot without raising an error. Catch it here.
    total_power = np.sum(Pxx_grand)
    if not np.isfinite(total_power) or total_power < 1e-20:
        print(f"    ⚠ Degenerate PSD for {condition_name}/{band_name} "
              f"(total power={total_power:.3e}, mean spike rate="
              f"{np.mean(spike_rates):.3f} Hz) — likely a silenced/"
              f"non-spiking network. Skipping this run.")
        return {"success": False}

    Pxx_rel = Pxx_grand / total_power

    if not np.all(np.isfinite(Pxx_rel)):
        print(f"    ⚠ Non-finite values in normalized PSD for "
              f"{condition_name}/{band_name} — skipping this run.")
        return {"success": False}

    return {
        "condition":  condition_name,
        "band":       band_name,
        "f":          f_out,
        "Pxx":        Pxx_rel,
        "spike_rate": np.mean(spike_rates),
        "success":    True,
    }


# ============================================================
# Bisection calibration of external drive (per condition)
# ============================================================
# Motivation: HC (g_ratio=6.5) and AD (g_ratio=2.5) have very different net
# inhibition, so a single fixed nu_ext either silences the more-inhibited
# condition or saturates the less-inhibited one (both observed already).
# Instead of guessing one global nu_ext, bisect it per condition on a short
# calibration run until mean E-population spike rate lands in a target
# band. Mirrors the bisection-based calibration approach already used for
# the PyNEST AD/HC E/I-imbalance runaway-firing problem.

def calibrate_nu_ext(condition_name,
                      g_ratio,
                      conn_matrix,
                      N_subnet,
                      N_subnets,
                      target_range=(3.0, 15.0),
                      lo=0.02,
                      hi=3.0,
                      max_iter=10,
                      calib_sim_time=5000.0,
                      calib_warmup=1000.0,
                      seed_base=42):
    """
    Bisect nu_ext for a given condition/g_ratio until mean E spike rate
    falls within target_range (Hz). calib_sim_time/calib_warmup default to
    the SAME values used in the production sweep -- using a shorter
    calibration window than production is misleading near a critical
    regime, where spikes only appear given enough time for the recurrent
    loop (and the slow Izhikevich recovery variable u) to build up
    activity. Returns the calibrated nu_ext and the rate it achieved.
    """
    target_lo, target_hi = target_range
    # Track "best" by (a) preferring any in-range result, then (b) among
    # all-zero results preferring the HIGHEST nu_ext tried (best chance of
    # eventually crossing threshold), rather than the first one tried.
    best_nu_ext = lo
    best_rate   = 0.0
    best_gap    = np.inf
    hit_ceiling = False

    print(f"\n  Calibrating nu_ext for {condition_name} (g_ratio={g_ratio})...")

    for it in range(max_iter):
        mid = (lo + hi) / 2
        result = run_simulation_with_fc(
            condition_name, "calib", g_ratio, conn_matrix,
            N_subnet=N_subnet, N_subnets=N_subnets,
            nu_ext=mid, sim_time=calib_sim_time, warmup=calib_warmup,
            seed_base=seed_base,
        )

        rate = result["spike_rate"] if result["success"] else 0.0
        in_range = target_lo <= rate <= target_hi
        gap = 0.0 if in_range else min(abs(rate - target_lo), abs(rate - target_hi))

        # Prefer strictly better gap; on ties, prefer the higher nu_ext
        # (relevant specifically for the "always 0 Hz" case above).
        if gap < best_gap or (gap == best_gap and mid > best_nu_ext):
            best_gap, best_rate, best_nu_ext = gap, rate, mid

        print(f"    iter {it+1:>2}: nu_ext={mid:.4f} -> rate={rate:.3f} Hz "
              f"(range [{lo:.4f}, {hi:.4f}])")

        if in_range:
            print(f"    ✓ Converged: nu_ext={mid:.4f} -> {rate:.3f} Hz")
            return mid, rate

        if rate < target_lo:
            lo = mid   # too quiet -> raise the floor
        else:
            hi = mid   # too saturated -> lower the ceiling

        if it == max_iter - 1 and rate < target_lo:
            hit_ceiling = True

    if hit_ceiling:
        print(f"    ⚠ Rate stayed below target even at nu_ext≈{hi:.4f} "
              f"(the upper bound tried). This network may be sitting at a "
              f"near-critical operating point where spiking is rare/slow "
              f"to emerge -- consider raising the search ceiling (hi) or "
              f"increasing g_E_base, rather than trusting this calibration.")

    print(f"    ⚠ Did not converge in {max_iter} iters; using closest "
          f"nu_ext={best_nu_ext:.4f} -> {best_rate:.3f} Hz")
    return best_nu_ext, best_rate


# ============================================================
# Stitch spectra
# ============================================================

def stitch_spectra(results_by_condition, band_ranges):
    stitched = {}
    for condition, band_results in results_by_condition.items():
        all_f, all_Pxx = [], []
        for band_name in ['delta', 'theta', 'alpha', 'beta', 'gamma']:
            if band_name not in band_results:
                continue
            result      = band_results[band_name]
            f, Pxx      = result['f'], result['Pxx']
            low, high   = band_ranges[band_name]
            mask        = (f >= low) & (f < high)
            all_f.append(f[mask])
            all_Pxx.append(Pxx[mask])

        if all_f:
            sf  = np.concatenate(all_f)
            sp  = np.concatenate(all_Pxx)
            sp /= np.sum(sp)
            stitched[condition] = {"f": sf, "Pxx": sp}
    return stitched


# ============================================================
# Parameters
# ============================================================

DATA_ROOT   = "../ds004504/"
BAND_RANGES = {
    "delta": (1,   4),
    "theta": (4,   8),
    "alpha": (8,  13),
    "beta":  (13, 30),
    "gamma": (30, 45),
}

CONDITIONS = {
    "HC": 6.5,
    "AD": 2.5,
}

N_SUBNETS = 19
N_SUBNET  = 263 #21
N_REPEATS = 10

# Per-band label list (shared across all print sections)
BANDS = [("Delta", 1, 4), ("Theta", 4, 8),
         ("Alpha", 8, 13), ("Beta", 13, 30), ("Gamma", 30, 40)]


# ============================================================
# Calibrate external drive per condition (bisection)
# ============================================================
# Target a modest, plausible cortical E-population firing rate band so
# neither condition is silent nor saturated. Uses the "alpha" band's
# connectivity as a representative matrix for calibration purposes only —
# the resulting nu_ext is then reused across all bands/repeats for that
# condition, so calibration cost is paid once per condition, not once per
# band/repeat.

print("=" * 60)
print("Calibrating external drive (nu_ext) per condition")
print("=" * 60)

CALIBRATED_NU_EXT = {}
for condition, g_ratio in CONDITIONS.items():
    conn = load_connectivity_matrix(condition, "alpha", DATA_ROOT)
    idx, subnet_conn = select_subnetworks(conn, N_SUBNETS)
    calibrated_nu_ext, calibrated_rate = calibrate_nu_ext(
        condition, g_ratio, subnet_conn,
        N_subnet=N_SUBNET, N_subnets=N_SUBNETS,
        target_range=(3.0, 15.0),
    )
    CALIBRATED_NU_EXT[condition] = calibrated_nu_ext

print("\nCalibrated nu_ext:")
for condition, nu in CALIBRATED_NU_EXT.items():
    print(f"  {condition}: nu_ext={nu:.4f}")


# ============================================================
# Repeated experiment
# ============================================================

stitched_runs = {c: [] for c in CONDITIONS}
f_axes        = {c: None for c in CONDITIONS}

# Per-condition, per-band spike rates (Hz) — collected so we can check
# whether the network is actually spiking, and whether AD vs HC diverge
# at the spiking level, before trusting the PSD comparison.
spike_rates_runs = {c: {b: [] for b in BAND_RANGES} for c in CONDITIONS}

print("Starting FC-based simulations")
print("=" * 60)
print(f"Running {N_REPEATS} repeats per condition")

for repeat in range(N_REPEATS):
    print("\n" + "=" * 70)
    print(f"REPEAT {repeat + 1}/{N_REPEATS}")
    print("=" * 70)

    results_by_condition = {c: {} for c in CONDITIONS}

    for condition, g_ratio in CONDITIONS.items():
        for band_name, (low, high) in BAND_RANGES.items():
            try:
                conn = load_connectivity_matrix(condition, band_name, DATA_ROOT)
                idx, subnet_conn = select_subnetworks(conn, N_SUBNETS)
                result = run_simulation_with_fc(
                    condition, band_name, g_ratio, subnet_conn,
                    N_subnet=N_SUBNET, N_subnets=N_SUBNETS,
                    nu_ext=CALIBRATED_NU_EXT[condition],
                    seed_base=42 + repeat * 100
                )
                if result["success"]:
                    results_by_condition[condition][band_name] = result
                    spike_rates_runs[condition][band_name].append(result["spike_rate"])
            except Exception as e:
                print(f"  Error ({condition}, {band_name}): {e}")

    stitched = stitch_spectra(results_by_condition, BAND_RANGES)

    for condition in stitched:
        stitched_runs[condition].append(stitched[condition]["Pxx"])
        f_axes[condition] = stitched[condition]["f"]

print("\n" + "=" * 60)
print(f"All repeats done.")


# ============================================================
# Spike-rate diagnostics — check BEFORE trusting the PSD comparison
# ============================================================
# If rates are near 0 Hz in both conditions, the "spectrum" above is coming
# from subthreshold membrane resonance (a function of delays / passive
# dynamics), not from spiking population activity, and it will look nearly
# identical across g_ratio values for that reason. If rates are enormous
# (hundreds/thousands of Hz), the network is saturated by the external
# drive rather than expressing E/I-dependent dynamics. Either way, AD vs
# HC won't diverge meaningfully in the PSD until rates are in a plausible
# cortical range (roughly 1-50 Hz) AND differ between conditions.

print("\n" + "=" * 60)
print("Spike-rate diagnostics (mean E-population rate, Hz)")
print("=" * 60)
print(f"  {'Condition':<10} {'Band':<8} {'n runs':>8}  {'Mean Hz':>10}  {'Std Hz':>10}")
print(f"  {'-'*52}")
for condition in CONDITIONS:
    for band_name in BAND_RANGES:
        rates = spike_rates_runs[condition][band_name]
        if rates:
            r = np.array(rates)
            print(f"  {condition:<10} {band_name:<8} {len(r):>8}  "
                  f"{r.mean():>10.3f}  {r.std(ddof=1) if len(r) > 1 else 0.0:>10.3f}")
        else:
            print(f"  {condition:<10} {band_name:<8} {'0':>8}  {'--':>10}  {'--':>10}")

all_rates = [r for c in CONDITIONS for b in BAND_RANGES for r in spike_rates_runs[c][b]]
if all_rates:
    arr = np.array(all_rates)
    if arr.mean() < 1.0:
        print("\n  ⚠ Mean spike rate across all runs is <1 Hz — the network is "
              "very likely subthreshold. The PSD comparison above is probably "
              "not meaningful yet; consider increasing nu_ext, g_E_base, or "
              "p_conn_local.")
    elif arr.mean() > 200.0:
        print("\n  ⚠ Mean spike rate across all runs is >200 Hz — the network "
              "is very likely saturated/synchronized by the external drive "
              "rather than expressing E/I-dependent dynamics; consider "
              "reducing nu_ext.")


# ============================================================
# Aggregate: mean, std, bootstrap CI
# ============================================================

stitched_stats = {}   # keyed by condition; holds mean, std, ci_low, ci_high, n

for condition in CONDITIONS:
    runs = stitched_runs[condition]
    if not runs:
        print(f"  No successful runs for {condition} — skipping.")
        continue

    # Interpolate all runs onto the first run's frequency grid
    f_ref     = f_axes[condition]
    powers    = []
    n_dropped = 0
    for pxx in runs:
        if len(pxx) == len(f_ref):
            candidate = pxx
        else:
            interp_fn = interp1d(
                np.linspace(f_ref[0], f_ref[-1], len(pxx)),
                pxx, kind='linear', bounds_error=False, fill_value='extrapolate'
            )
            candidate = interp_fn(f_ref)

        if not np.all(np.isfinite(candidate)):
            n_dropped += 1
            continue
        powers.append(candidate)

    if n_dropped:
        print(f"  ⚠ {condition}: dropped {n_dropped}/{len(runs)} run(s) "
              f"with non-finite power values before aggregation.")

    if not powers:
        print(f"  No finite runs for {condition} — skipping.")
        continue

    powers_array = np.array(powers)            # (n_runs, n_freqs)
    n_ok         = powers_array.shape[0]

    mean_power = np.mean(powers_array, axis=0)
    std_power  = np.std( powers_array, axis=0, ddof=1)

    ci_low, ci_high = bootstrap_ci(
        powers_array, n_boot=2000, alpha=0.05,
        rng_seed=int(10 * CONDITIONS[condition])
    )

    stitched_stats[condition] = {
        "f":        f_ref,
        "mean":     mean_power,
        "std":      std_power,
        "ci_low":   ci_low,
        "ci_high":  ci_high,
        "n":        n_ok,
    }

    print(f"\n  ✓ {condition}: {n_ok}/{N_REPEATS} repeats  "
          f"|  95 % CI computed (bootstrap, n_boot=2000)")


# ============================================================
# Print 95 % bootstrapped CI per condition and band
# ============================================================

print("\n" + "=" * 60)
print("95 % Bootstrapped Confidence Intervals (per band)")
print("=" * 60)

for condition, s in stitched_stats.items():
    print(f"\n  Condition : {condition}  "
          f"(g_I/g_E = {CONDITIONS[condition]},  n = {s['n']})")
    print(f"  {'Band':<10} {'Freq (Hz)':<14} {'Mean power':>12}  "
          f"{'CI low':>12}  {'CI high':>12}")
    print(f"  {'-'*62}")
    f = s['f']
    for bname, flo, fhi in BANDS:
        mask = (f >= flo) & (f < fhi)
        if mask.any():
            mean_b = s['mean'][mask].mean()
            ci_lo  = s['ci_low'][mask].mean()
            ci_hi  = s['ci_high'][mask].mean()
            print(f"  {bname:<10} {flo:>2}–{fhi:<5} Hz    "
                  f"{mean_b:>12.6f}  {ci_lo:>12.6f}  {ci_hi:>12.6f}")


# ============================================================
# Cohen's d (pooled SD) — AD vs HC
# ============================================================

if 'AD' in stitched_stats and 'HC' in stitched_stats:
    ad = stitched_stats['AD']
    hc = stitched_stats['HC']

    # Interpolate HC onto AD's frequency grid if they differ
    f_ref = ad['f']
    if not np.allclose(ad['f'], hc['f']):
        for key in ('mean', 'std', 'ci_low', 'ci_high'):
            interp_fn  = interp1d(hc['f'], hc[key], kind='linear',
                                  bounds_error=False, fill_value='extrapolate')
            hc[key] = interp_fn(f_ref)

    d_spectrum, s_pooled = cohens_d_pooled(
        mu1=ad['mean'], sd1=ad['std'], n1=ad['n'],
        mu2=hc['mean'], sd2=hc['std'], n2=hc['n'],
    )

    abs_d    = np.abs(d_spectrum)
    peak_idx = np.argmax(abs_d)

    print("\n" + "=" * 60)
    print("Cohen's d (pooled SD) — AD vs HC")
    print("=" * 60)
    print(f"  Sign convention         : positive d = AD > HC")
    print(f"  Mean |d| across 1–40 Hz : {abs_d.mean():.4f}")
    print(f"  Peak |d|                : {abs_d[peak_idx]:.4f}  "
          f"at {f_ref[peak_idx]:.1f} Hz")

    print(f"\n  {'Band':<10} {'Freq (Hz)':<14} {'mean_AD':>10}  {'mean_HC':>10}  "
          f"{'s_pooled':>10}  {'Cohen d':>10}  {'|d|':>8}")
    print(f"  {'-'*80}")

    for bname, flo, fhi in BANDS:
        mask = (f_ref >= flo) & (f_ref < fhi)
        if mask.any():
            m_ad  = ad['mean'][mask].mean()
            m_hc  = hc['mean'][mask].mean()
            sp    = s_pooled[mask].mean()
            d_val = d_spectrum[mask].mean()
            print(f"  {bname:<10} {flo:>2}–{fhi:<5} Hz    "
                  f"{m_ad:>10.6f}  {m_hc:>10.6f}  {sp:>10.6f}  "
                  f"{d_val:>10.4f}  {abs(d_val):>8.4f}")

else:
    d_spectrum = None
    f_ref      = None
    print("\n  Skipping Cohen's d (need both AD and HC results)")


# ============================================================
# Plotting — mean ± std shading per condition
# ============================================================

colors = {
    "AD":  "#90EE90",
    "HC":  "#A9A9A9",
}

if stitched_stats:
    fig, ax = plt.subplots(figsize=(10, 6))

    for condition, s in stitched_stats.items():
        f, mean, std = s['f'], s['mean'], s['std']

        # Smooth in log-domain for display only; raw mean/std (above) are
        # untouched and still used for the band table / Cohen's d earlier.
        mean_smooth = smooth_log_domain(mean)
        std_smooth  = smooth_log_domain(std)

        ax.plot(f, mean_smooth,
                color=colors.get(condition, None),
                linewidth=2.5,
                label=condition,
                alpha=0.85)
        ax.fill_between(f, mean_smooth - std_smooth, mean_smooth + std_smooth,
                        color=colors.get(condition, None),
                        alpha=0.25)

    band_boundaries = [4, 8, 13, 30]
    for bnd in band_boundaries:
        ax.axvline(x=bnd, color='gray', linestyle='--', linewidth=1.5, alpha=0.6)

    y_max        = ax.get_ylim()[1]
    band_centers = [(1+4)/2, (4+8)/2, (8+13)/2, (13+30)/2, (30+40)/2]
    band_names   = ['Delta', 'Theta', 'Alpha', 'Beta', 'Gamma']
    for center, bname in zip(band_centers, band_names):
        if center <= 40:
            ax.text(center, y_max * 0.7, bname,
                    ha='center', fontsize=9, style='italic',
                    color='gray', alpha=0.7)

    ax.set_xlim(1, 40)
    ax.set_xlabel("Frequency (Hz)", fontsize=14, fontweight='bold')
    ax.set_ylabel("Relative power",  fontsize=14, fontweight='bold')
    ax.legend(fontsize=10, loc='upper right')
    ax.grid(alpha=0.3)
    ax.set_title("Power spectra — mean ± SD, log-domain Savitzky-Golay smoothed  "
                 "(FC-driven network, Izhikevich neurons)", fontsize=13)

    plt.tight_layout()
    plt.savefig("stitched_spectrum_mean_std.png", dpi=300)
    plt.savefig("stitched_spectrum_mean_std.pdf")
    plt.close()

    print("\n✓ Plots saved:")
    print("  - stitched_spectrum_mean_std.png")
    print("  - stitched_spectrum_mean_std.pdf")

print("\nDone.")
