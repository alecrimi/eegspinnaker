# ============================================================
# PyNEST – Improved E/I imbalance with FC-based connectivity
# IZHIKEVICH NEURON VERSION — BROADBAND (single-network) revision
#
# === MAJOR STRUCTURAL CHANGE (this revision) ===
# Previous versions simulated FIVE separate band-specific networks
# per condition (one per delta/theta/alpha/beta/gamma FC matrix),
# each independently calibrated to a target rate, then crossfaded
# ("stitched") together into one spectrum. That produced visible
# seams / amplitude discontinuities at band boundaries, because
# five independently-tuned simulations don't have any reason to
# agree on amplitude where they meet.
#
# This version runs ONE simulation per condition, using the ALPHA
# connectivity matrix as the (single) structural backbone, and
# takes its full 1-40 Hz Welch PSD directly. No stitching.
#
# === SPECTRAL TILT (this revision, replaces an earlier time-domain
# synaptic filter that produced a hard ~10 Hz low-pass cutoff) ===
# The population-averaged V_m "LFP proxy" has a spectral background
# close to flat/white (delta-kick synapses + point-process-like
# spiking have no built-in low-pass stage), unlike real LFP/EEG which
# gets an aperiodic 1/f-like background from synaptic + dendritic
# filtering. This is approximated here with a frequency-domain
# power-law tilt (Pxx /= f^exponent) applied directly to the PSD
# AFTER Welch. This is a POST-HOC reshaping of the spectrum only --
# it does not change network dynamics, spike rates, or calibration,
# and (unlike a time-domain low-pass kernel) it doesn't impose a hard
# corner frequency that crushes beta/gamma to zero.
#
# === CALIBRATION (this revision) ===
# nu_ext is now calibrated ONCE PER CONDITION (on the alpha FC
# matrix, matching what's actually simulated), instead of once
# per (condition, band) pair. AD and HC are each calibrated on
# their OWN g_ratio (this was fixed in the previous revision and
# is preserved here) -- AD no longer inherits HC's drive.
# ============================================================

import nest
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch, savgol_filter
import os

# ============================================================
# Constants
# ============================================================
EXTERNAL_SCALE = 250.0
P_INTER_BASE   = 0.2
TARGET_RATE    = 8.0
TOLERANCE      = 0.25

# === RECURRENT GAIN (tuned toward the resonance band) ===
# G_E_RECURRENT and coupling_strength (passed separately, search for
# "coupling_strength=") together set the gain of the delayed E->I->E
# feedback loop. Bisection progress so far:
#   - G=2.0, coupling=1.5   -> hard limit cycle (sharp peak+harmonics)
#   - G=1.2, coupling=0.8   -> overdamped (no peak, smooth 1/f decay)
#   - G=1.5, coupling=1.1   -> weak broad bumps, HC/AD at different
#     frequencies (g_ratio-driven, expected)
#   - G=1.7, coupling=1.3   -> visible alpha resonance for both, BUT
#     AD's peak comes out narrower/taller than HC's at nearly-tied
#     band-integrated power (Cohen's d ~ -0.11 in 8-13 Hz -- i.e. not
#     really "AD > HC", just AD's resonance is more CONCENTRATED).
#     This happens because AD's much weaker inhibition (g_ratio=2.5
#     vs HC's 6.5) puts it systematically closer to the oscillatory
#     threshold at any SHARED gain level, so AD sharpens faster than
#     HC as gain increases. Backing off slightly to pull AD back from
#     threshold a bit more (HC has more margin already, so it should
#     lose comparatively less resonance from the same gain decrease).
# NOTE: g_ratio (g_I/g_E, i.e. CONDITIONS['HC']/['AD']) is deliberately
# left untouched -- that's the actual HC vs AD experimental variable.
G_E_RECURRENT  = 1.6

WEIGHT_JITTER_FRAC       = 0.15
DELAY_JITTER_FRAC_LOCAL  = 0.33   # ~= 0.5ms / 1.5ms delay_local, expressed as a fraction
DELAY_JITTER_FRAC_INTER  = 0.67   # ~= 2.0ms / 3.0ms delay_inter, expressed as a fraction

A_JITTER_STD_FRAC = 0.3
B_JITTER_STD_FRAC = 0.1

REFERENCE_N_PER_REGION = 21

# === JITTER SCALING (new) ===
# The desynchronization jitter values above (WEIGHT_JITTER_FRAC,
# DELAY_JITTER_MS, A_JITTER_STD_FRAC, etc.) were tuned to be "just
# enough" independent per-neuron/per-synapse noise to mask an
# underlying ~3.5 Hz mean-field resonance in the population-averaged
# LFP proxy, AT N_per_region = REFERENCE_N_PER_REGION (21).
#
# That masking noise is independent across neurons/synapses, so its
# contribution to the POPULATION AVERAGE shrinks as ~1/sqrt(N) (law
# of large numbers) as N_per_region grows -- while the underlying
# mean-field oscillatory tendency does NOT shrink with N. So at large
# N_per_region, the noise that used to hide the resonance is no
# longer strong enough, and the averaged LFP proxy converges toward
# the bare mean-field oscillation: sharp, narrow peaks at ~3.5/7/10.5
# Hz (a fundamental + harmonics), instead of the broad smoothed bumps
# seen at N=21.
#
# Fix: scale every jitter magnitude by sqrt(N_per_region /
# REFERENCE_N_PER_REGION) so the EFFECTIVE noise level in the
# population average stays roughly constant regardless of N. Capped
# at MAX_JITTER_SCALE to avoid weights/delays becoming degenerate
# (sign-flipping, near-zero delay) at very large N.
MAX_JITTER_SCALE = 4.0

def get_jitter_scale(n_per_region):
    """sqrt-N scaling factor so population-average desync noise stays
    roughly constant as N_per_region grows away from the reference."""
    raw_scale = np.sqrt(n_per_region / REFERENCE_N_PER_REGION)
    return min(raw_scale, MAX_JITTER_SCALE)

# === SPECTRAL TILT CONSTANT (replaces the time-domain synaptic filter) ===
# The previous alpha-kernel time-domain filter (t*exp(-t/tau)) is a
# second-order low-pass with corner frequency ~= 1/(2*pi*tau) -- at
# tau=15ms that's ~10.6 Hz, which produced a HARD cutoff (everything
# above ~15 Hz crushed near zero) rather than a gentle 1/f-style tilt.
# It also introduced a convolution edge artifact at the start of the
# trace (the negative dip near 1 Hz).
#
# Replaced with a frequency-domain power-law tilt applied directly to
# the PSD after Welch: Pxx_shaped(f) = Pxx(f) / f^SPECTRAL_TILT_EXPONENT.
# This gives a smooth, monotonic 1/f^n-style decay across the WHOLE
# 1-40 Hz range with no hard corner and no time-domain edge effects.
# Larger exponent -> steeper decay (more low-frequency-dominated).
# Typical cortical LFP/EEG aperiodic exponents are roughly 1-3;
# start around 1.0-1.5 and tune by eye.
SPECTRAL_TILT_EXPONENT = 1.2

# === SPIKE-CLIP CONSTANT (new) ===
# Ceiling (mV) applied to V_m before averaging into the LFP proxy, to
# exclude spike depolarization/reset transients (see full explanation
# at the clipping call site in run_simulation_with_fc). RS/FS resting
# baseline in this model is ~-65 to -70 mV; -40 mV sits comfortably
# below spike threshold (~+30 mV to trigger, per Izhikevich dynamics)
# while still allowing normal subthreshold depolarization through.
# Lower it (e.g. -50) for more aggressive exclusion of near-threshold
# activity, raise it (e.g. -30) if it's clipping too much legitimate
# subthreshold signal.
SPIKE_CLIP_MV = -40.0


# ============================================================
# Bootstrap CI helper
# ============================================================

def bootstrap_ci(powers_array, n_boot=2000, alpha=0.05, rng_seed=0):
    """Per-bin bootstrapped confidence interval. powers_array: (n_runs, n_freqs)."""
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
# Cohen's d (pooled SD) helper
# ============================================================

def cohens_d_pooled(mu1, sd1, n1, mu2, sd2, n2):
    """Per-bin Cohen's d using pooled SD."""
    mu1, sd1 = np.asarray(mu1), np.asarray(sd1)
    mu2, sd2 = np.asarray(mu2), np.asarray(sd2)
    s_pooled = np.sqrt(
        ((n1 - 1) * sd1**2 + (n2 - 1) * sd2**2) / (n1 + n2 - 2)
    )
    with np.errstate(invalid='ignore', divide='ignore'):
        d = np.where(s_pooled > 0, (mu1 - mu2) / s_pooled, 0.0)
    return d, s_pooled


# ============================================================
# Per-band label list (used ONLY for summary statistics printing —
# the simulation itself is broadband now, this just buckets the
# resulting spectrum for reporting)
# ============================================================

BANDS = [("Delta", 1, 4), ("Theta", 4, 8),
         ("Alpha", 8, 13), ("Beta", 13, 30), ("Gamma", 30, 40)]


# ============================================================
# Load connectivity
# ============================================================

def load_connectivity_matrix(group, band, data_root="./"):
    """Load functional connectivity matrix from PLV data."""
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

def select_regions(conn_matrix, n_regions=19):
    """Select regions with highest connectivity."""
    node_strength    = np.sum(np.abs(conn_matrix), axis=1)
    selected_indices = np.argsort(node_strength)[-n_regions:]
    region_conn      = conn_matrix[np.ix_(selected_indices, selected_indices)]
    return selected_indices, region_conn


# ============================================================
# Per-neuron heterogeneous Poisson input
# ============================================================

def create_heterogeneous_poisson_inputs(n_neurons, base_rate_hz, sigma=0.5):
    """One Poisson generator per neuron, rate ~ lognormal around base_rate_hz."""
    rates = base_rate_hz * np.random.lognormal(mean=0.0, sigma=sigma, size=n_neurons)
    rates = np.clip(rates, 0.1 * base_rate_hz, 10.0 * base_rate_hz)
    generators = []
    for rate in rates:
        pg = nest.Create("poisson_generator", 1, {"rate": float(rate)})
        generators.append(pg)
    return generators


# ============================================================
# Per-synapse jitter helpers
# ============================================================

def jittered_weight(mean_weight, frac=WEIGHT_JITTER_FRAC):
    lo, hi = sorted([mean_weight * (1.0 - frac), mean_weight * (1.0 + frac)])
    return nest.random.uniform(min=lo, max=hi)


def jittered_delay(mean_delay, frac=DELAY_JITTER_FRAC_LOCAL):
    """
    Per-synapse uniformly-jittered delay, drawn from
    [mean_delay*(1-frac), mean_delay*(1+frac)] -- same fractional
    (multiplicative) approach as jittered_weight(), rather than an
    absolute ms half-width.
    NOTE: this replaces an earlier absolute-ms version. That version
    floored the lower bound at 0.1ms; once the (N-scaled) jitter
    half-width exceeded mean_delay, the lower bound kept hitting that
    floor while the upper bound kept growing, which silently biased
    the MEAN delay upward as the jitter scale grew with N -- exactly
    the opposite of what "N-invariant desync noise" was supposed to
    do. The fractional form here can't produce that bias as long as
    frac < 1 (mean_delay*(1-frac) stays positive by construction), so
    frac is capped at a safe value wherever it's scaled with N.
    """
    lo, hi = sorted([mean_delay * (1.0 - frac), mean_delay * (1.0 + frac)])
    # Floor raised from the original 0.1ms: at high jitter fractions (frac
    # approaching 0.9, which happens automatically at large N_per_region
    # via jscale), the lower bound can get very small (e.g. ~0.15-0.3ms),
    # which can make NEST's network-wide minimum delay awkwardly tiny and
    # trigger its "simulation time not a multiple of minimal delay"
    # warning. 0.5ms keeps delays in a sane physiological range and
    # avoids that. This floor is hit rarely and symmetrically (only at
    # the extreme low end of a wide jitter range), so it does not
    # reintroduce the old additive-jitter mean-drift bug.
    lo = max(0.5, lo)
    return nest.random.uniform(min=lo, max=hi)


# ============================================================
# Spectral tilt: post-hoc frequency-domain power-law shaping (NEW)
# ============================================================

def apply_spectral_tilt(f, Pxx, exponent=SPECTRAL_TILT_EXPONENT):
    """
    Divide the PSD by f^exponent to impose a smooth 1/f^n-style decay
    across the whole band, approximating the aperiodic background that
    synaptic/dendritic filtering gives real LFP/EEG -- without the hard
    corner-frequency cutoff a time-domain low-pass kernel would impose,
    and without any convolution edge artifacts. Applied AFTER Welch, to
    the PSD directly, not to the raw LFP trace.
    """
    f_safe = np.maximum(f, 1.0)   # avoid blow-up as f -> 0
    return Pxx / (f_safe ** exponent)


# ============================================================
# Single simulation (BROADBAND — one network per condition)
# ============================================================

GLOBAL_MAX = None   # assigned in main, from HC/AD alpha FC

def run_simulation_with_fc(condition,
                           g_ratio,
                           conn_matrix,
                           N_per_region=21,
                           n_regions=19,
                           frac_exc=0.8,
                           p_conn_local=0.1,
                           coupling_strength=1.2,
                           nu_ext=0.03,
                           sim_time=10000.0,
                           warmup=3000.0,
                           seed=42,
                           record_fraction=0.2,
                           smooth_spectrum=True,
                           poisson_sigma=0.5):
    """
    Run E/I network (Izhikevich neurons) with FC-weighted inter-region
    coupling, using a SINGLE structural connectivity matrix (alpha FC)
    for the whole 1-40 Hz PSD. No per-band re-simulation, no stitching.
    """
    nest.ResetKernel()
    nest.resolution = 0.1
    nest.set_verbosity("M_WARNING")

    rng_seed = seed + hash(condition) % 1000
    np.random.seed(rng_seed)
    nest.SetKernelStatus({"rng_seed": rng_seed})

    print(f"\n[{condition}] g_I/g_E = {g_ratio:.2f}, seed = {rng_seed}")

    N_E_region = int(N_per_region * frac_exc)
    N_I_region = N_per_region - N_E_region

    N_E_ref = int(REFERENCE_N_PER_REGION * frac_exc)
    N_I_ref = REFERENCE_N_PER_REGION - N_E_ref
    scale_E = N_E_ref / N_E_region
    scale_I = N_I_ref / N_I_region

    # === JITTER SCALING: effective (N-adjusted) desync magnitudes ===
    # so population-averaged desync noise stays roughly constant as
    # N_per_region grows away from the reference (see get_jitter_scale
    # docstring above).
    jscale = get_jitter_scale(N_per_region)
    a_jitter_eff = min(0.9, A_JITTER_STD_FRAC * jscale)
    b_jitter_eff = min(0.9, B_JITTER_STD_FRAC * jscale)
    weight_jitter_frac_eff = min(0.9, WEIGHT_JITTER_FRAC * jscale)
    delay_jitter_frac_local_eff = min(0.9, DELAY_JITTER_FRAC_LOCAL * jscale)
    delay_jitter_frac_inter_eff = min(0.9, DELAY_JITTER_FRAC_INTER * jscale)
    if N_per_region != REFERENCE_N_PER_REGION:
        print(f"    Jitter scale (N_per_region={N_per_region}): {jscale:.2f}x "
              f"(weight_frac={weight_jitter_frac_eff:.3f}, "
              f"delay_local_frac={delay_jitter_frac_local_eff:.3f}, "
              f"delay_inter_frac={delay_jitter_frac_inter_eff:.3f}, "
              f"a_frac={a_jitter_eff:.3f}, b_frac={b_jitter_eff:.3f})")

    RS_A, RS_B = 0.02, 0.2
    FS_A, FS_B = 0.1,  0.2

    E_regions = []
    I_regions = []

    for i in range(n_regions):
        E = nest.Create("izhikevich", N_E_region)
        I = nest.Create("izhikevich", N_I_region)

        E.a = np.clip(RS_A + a_jitter_eff * RS_A * np.random.randn(N_E_region),
                      0.008, 0.04)
        E.b = np.clip(RS_B + b_jitter_eff * RS_B * np.random.randn(N_E_region),
                      0.15, 0.25)
        E.c = -65.0 + 5.0 * np.random.randn(N_E_region)
        E.d = 8.0 + 1.0 * np.random.randn(N_E_region)

        I.a = np.clip(FS_A + a_jitter_eff * FS_A * np.random.randn(N_I_region),
                      0.05, 0.2)
        I.b = np.clip(FS_B + b_jitter_eff * FS_B * np.random.randn(N_I_region),
                      0.15, 0.25)
        I.c = -65.0 + 5.0 * np.random.randn(N_I_region)
        I.d = 2.0 + 0.5 * np.random.randn(N_I_region)

        E.V_m = -70.0 + 5.0 * np.random.randn(N_E_region)
        I.V_m = -70.0 + 5.0 * np.random.randn(N_I_region)
        E.U_m = np.array(E.b) * np.array(E.V_m)
        I.U_m = np.array(I.b) * np.array(I.V_m)

        E_regions.append(E)
        I_regions.append(I)

    g_E         = G_E_RECURRENT
    g_I         = g_ratio * g_E
    delay_local = 1.5
    delay_inter = 3.0

    base_rate_hz = nu_ext * EXTERNAL_SCALE
    for region_idx in range(n_regions):
        ext_E_gens = create_heterogeneous_poisson_inputs(
            N_E_region, base_rate_hz, sigma=poisson_sigma)
        ext_I_gens = create_heterogeneous_poisson_inputs(
            N_I_region, base_rate_hz, sigma=poisson_sigma)

        for pg, neuron in zip(ext_E_gens, E_regions[region_idx]):
            nest.Connect(pg, neuron, syn_spec={"weight": g_E, "delay": delay_local})
        for pg, neuron in zip(ext_I_gens, I_regions[region_idx]):
            nest.Connect(pg, neuron, syn_spec={"weight": 1.3 * g_E, "delay": delay_local})

    p_from_E = min(1.0, p_conn_local * scale_E)
    p_from_I = min(1.0, p_conn_local * scale_I)
    conn_from_E = {"rule": "pairwise_bernoulli", "p": p_from_E}
    conn_from_I = {"rule": "pairwise_bernoulli", "p": p_from_I}

    for region_idx in range(n_regions):
        E = E_regions[region_idx]
        I = I_regions[region_idx]
        nest.Connect(E, E, conn_from_E, syn_spec={
            "weight": jittered_weight(g_E, frac=weight_jitter_frac_eff),
            "delay":  jittered_delay(delay_local, frac=delay_jitter_frac_local_eff)})
        nest.Connect(E, I, conn_from_E, syn_spec={
            "weight": jittered_weight(1.2 * g_E, frac=weight_jitter_frac_eff),
            "delay":  jittered_delay(delay_local, frac=delay_jitter_frac_local_eff)})
        nest.Connect(I, E, conn_from_I, syn_spec={
            "weight": jittered_weight(-g_I, frac=weight_jitter_frac_eff),
            "delay":  jittered_delay(delay_local, frac=delay_jitter_frac_local_eff)})
        nest.Connect(I, I, conn_from_I, syn_spec={
            "weight": jittered_weight(-0.8 * g_I, frac=weight_jitter_frac_eff),
            "delay":  jittered_delay(delay_local, frac=delay_jitter_frac_local_eff)})

    if GLOBAL_MAX is not None and GLOBAL_MAX > 0:
        conn_normalized = conn_matrix / GLOBAL_MAX
    else:
        conn_normalized = conn_matrix / np.max(conn_matrix) if np.max(conn_matrix) > 0 else conn_matrix

    for i in range(n_regions):
        for j in range(n_regions):
            if i == j:
                continue
            fc_weight = max(conn_normalized[i, j], 0.0)
            if fc_weight > 1e-6:
                p_inter = min(P_INTER_BASE * fc_weight, 0.30) * scale_E
                p_inter = min(p_inter, 1.0)
                inter_weight = coupling_strength * g_E
                nest.Connect(
                    E_regions[i], E_regions[j],
                    {"rule": "pairwise_bernoulli", "p": p_inter},
                    syn_spec={
                        "weight": jittered_weight(inter_weight, frac=weight_jitter_frac_eff),
                        "delay":  jittered_delay(delay_inter, frac=delay_jitter_frac_inter_eff),
                    }
                )

    # === RECORDING POOL FIX (new) ===
    # Previously n_rec_per_region = record_fraction * N_E_region, so the
    # number of neurons averaged into the LFP proxy GREW proportionally
    # with N_per_region (e.g. ~16 at N=21, ~32 at N=200, ~64 at N=400).
    # That's a second, independent noise-cancelling effect on top of
    # per-synapse jitter -- growing the recorded pool shrinks the
    # population-average's variance via plain law-of-large-numbers
    # averaging, regardless of how much per-neuron heterogeneity is
    # injected. The jitter-scaling fix only compensated for the latter,
    # not this. Fix: size the recorded pool off the REFERENCE population
    # (N_per_region=21), not the actual (possibly much larger) one, so
    # the LFP proxy's sample size -- and thus its intrinsic averaging
    # noise level -- stays roughly constant regardless of N_per_region.
    target_rec = max(21, int(record_fraction * N_E_ref))
    n_rec_per_region = min(target_rec, N_E_region)

    mm = nest.Create("multimeter")
    mm.set({"interval": 1.0, "record_from": ["V_m"]})
    for r in range(n_regions):
        nest.Connect(mm, E_regions[r][:n_rec_per_region])
    print(f"    Recording from {n_rec_per_region}/{N_E_region} neurons per region "
          f"({100*n_rec_per_region/N_E_region:.1f}%)")

    spike_rec = nest.Create("spike_recorder")
    for E in E_regions:
        nest.Connect(E, spike_rec)

    nest.Simulate(warmup)
    nest.Simulate(sim_time)

    ev      = mm.get("events")
    times   = np.array(ev["times"])
    senders = np.array(ev["senders"])
    V_m     = np.array(ev["V_m"])

    mask             = times > warmup
    times_filtered   = times[mask]
    senders_filtered = senders[mask]
    V_m_filtered     = V_m[mask]

    if len(times_filtered) < 1000:
        print(f"    WARNING: Insufficient data ({len(times_filtered)} points)")
        return {'success': False}

    unique_times   = np.sort(np.unique(times_filtered))
    unique_neurons = np.sort(np.unique(senders_filtered))
    n_times        = len(unique_times)
    n_neurons      = n_regions * n_rec_per_region

    print(f"    Recording from {n_neurons} neurons over {n_times} time points")

    expected_length = n_times * n_neurons
    if len(times_filtered) != expected_length:
        print(f"    WARNING: Expected {expected_length} points, got {len(times_filtered)}")
        V_m_matrix    = np.full((n_times, n_neurons), np.nan)
        neuron_to_idx = {nid: idx for idx, nid in enumerate(unique_neurons)}
        time_to_idx   = {t:   idx for idx, t   in enumerate(unique_times)}
        for i in range(len(times_filtered)):
            t_idx = time_to_idx[times_filtered[i]]
            n_idx = neuron_to_idx[senders_filtered[i]]
            V_m_matrix[t_idx, n_idx] = V_m_filtered[i]
    else:
        V_m_matrix = V_m_filtered.reshape(n_times, n_neurons)

    # === SPIKE-TRANSIENT CLIPPING (new) ===
    # Izhikevich neurons have a large, nearly-identical voltage
    # excursion on every spike (threshold ~+30 mV crashing down to
    # reset value c ~-65 mV) -- far bigger than the few-mV subthreshold
    # synaptic fluctuations an LFP proxy is meant to capture. Averaging
    # RAW V_m across the population means that whenever even a modest
    # fraction of neurons fire in the same few-ms window (which will
    # always happen to *some* degree in a recurrently-connected
    # network, regardless of synaptic weight/delay jitter), their huge
    # spike transients dominate the population average and imprint a
    # sharp peak at the population's dominant firing rhythm -- this is
    # a property of averaging raw V_m, not something per-synapse
    # heterogeneity can fix, since it doesn't touch spike waveform
    # shape at all.
    # Fix: clip V_m to a ceiling below spike threshold before
    # averaging, so the LFP proxy reflects subthreshold synaptic drive
    # (closer to what a real LFP/EEG signal is dominated by) instead of
    # being swamped by spike depolarization/reset transients.
    V_m_matrix = np.clip(V_m_matrix, a_min=None, a_max=SPIKE_CLIP_MV)

    lfp  = np.nanmean(V_m_matrix, axis=1)
    lfp -= np.nanmean(lfp)

    print(f"    LFP (pop.-avg V_m proxy, spike-clipped at {SPIKE_CLIP_MV} mV): "
          f"{len(lfp)} samples, "
          f"range [{np.nanmin(lfp):.2f}, {np.nanmax(lfp):.2f}] mV")

    fs      = 1000.0
    nperseg = min(len(lfp) // 2, 16384)
    nperseg = max(nperseg, min(8192, len(lfp)))

    f, Pxx = welch(lfp, fs=fs, nperseg=nperseg, noverlap=int(15 * nperseg // 16),
                   window='hann', detrend='linear')

    band = (f >= 1) & (f <= 40)
    f    = f[band]
    Pxx  = Pxx[band]

    if smooth_spectrum and len(Pxx) > 10:
        window_length = min(21, len(Pxx) // 2 * 2 - 1)
        if window_length >= 10:
            Pxx = savgol_filter(Pxx, window_length=window_length, polyorder=3)

    Pxx  = np.maximum(Pxx, 0)

    # === SPECTRAL TILT (new): impose a smooth 1/f^n aperiodic background
    # on the PSD directly, post-Welch/post-smoothing. Applied here (not
    # to the raw LFP) so it can't introduce time-domain edge artifacts
    # and doesn't risk a hard low-pass corner cutting off beta/gamma. ===
    Pxx = apply_spectral_tilt(f, Pxx, exponent=SPECTRAL_TILT_EXPONENT)
    Pxx = np.maximum(Pxx, 0)

    total_power = Pxx.sum()
    if not np.isfinite(total_power) or total_power < 1e-20:
        print(f"    WARNING: Degenerate PSD (total power={total_power:.3e}) "
              f"-- likely silenced/non-spiking network. Skipping this run.")
        return {'success': False}

    # normalize to relative power (as before)
    Pxx = Pxx / total_power

    spikes      = spike_rec.get("events")
    spike_times = spikes["times"]
    spike_times = spike_times[spike_times > warmup]
    total_neurons = n_regions * N_E_region
    spike_rate    = len(spike_times) / (total_neurons * sim_time / 1000.0)
    print(f"    Spike rate: {spike_rate:.2f} Hz, Welch: nperseg={nperseg}")

    return {
        "condition":  condition,
        "g_ratio":    g_ratio,
        "f":          f,
        "Pxx":        Pxx,
        "lfp":        lfp,
        "spike_rate": spike_rate,
        "success":    True,
    }


# ============================================================
# Bisection calibration of external drive
# ============================================================

def calibrate_nu_ext(condition,
                      g_ratio,
                      conn_matrix,
                      N_per_region,
                      n_regions,
                      coupling_strength,
                      target_rate=TARGET_RATE,
                      tolerance=TOLERANCE,
                      max_iter=20,
                      calib_sim_time=10000.0,
                      calib_warmup=3000.0,
                      seed=42):
    """Find nu_ext that yields mean E spike rate ~= target_rate."""
    print(f"\n  Calibrating nu_ext for {condition} (g_ratio={g_ratio}, "
          f"coupling_strength={coupling_strength:.3f}, seed={seed})...")

    lo = 0.05
    hi = None
    rate_lo = None
    rate_hi = None

    nu = lo
    for expand_iter in range(10):
        result = run_simulation_with_fc(
            condition, g_ratio, conn_matrix,
            N_per_region=N_per_region, n_regions=n_regions,
            coupling_strength=coupling_strength,
            nu_ext=nu, sim_time=calib_sim_time, warmup=calib_warmup,
            seed=seed, smooth_spectrum=False,
        )
        rate = result["spike_rate"] if result.get("success", False) else 0.0
        print(f"    expand: nu_ext={nu:.4f} -> rate={rate:.3f} Hz")
        if rate >= target_rate:
            hi = nu
            rate_hi = rate
            break
        lo = nu
        rate_lo = rate
        nu *= 2.0
    else:
        print(f"    (!) Even after expansion, rate stayed below target. "
              f"Using nu_ext={nu:.4f} (rate={rate:.3f} Hz) as best guess.")
        return nu, rate

    best_nu_ext = lo
    best_rate   = rate_lo if rate_lo is not None else 0.0
    best_gap    = abs(best_rate - target_rate)

    print(f"    Bracket found: lo={lo:.4f} (rate={rate_lo:.3f}), hi={hi:.4f} (rate={rate_hi:.3f})")
    for it in range(max_iter):
        mid = (lo + hi) / 2.0
        result = run_simulation_with_fc(
            condition, g_ratio, conn_matrix,
            N_per_region=N_per_region, n_regions=n_regions,
            coupling_strength=coupling_strength,
            nu_ext=mid, sim_time=calib_sim_time, warmup=calib_warmup,
            seed=seed, smooth_spectrum=False,
        )
        rate = result["spike_rate"] if result.get("success", False) else 0.0
        gap = abs(rate - target_rate)

        if gap < best_gap or (gap == best_gap and mid > best_nu_ext):
            best_gap, best_rate, best_nu_ext = gap, rate, mid

        print(f"    bisect iter {it+1:>2}: nu_ext={mid:.4f} -> rate={rate:.3f} Hz "
              f"(gap={gap:.3f})")

        if gap < tolerance:
            print(f"    Converged: nu_ext={mid:.4f} -> {rate:.3f} Hz")
            return mid, rate

        if rate < target_rate:
            lo = mid
            rate_lo = rate
        else:
            hi = mid
            rate_hi = rate

    print(f"    (!) Did not converge in {max_iter} iters; using closest "
          f"nu_ext={best_nu_ext:.4f} -> {best_rate:.3f} Hz")
    return best_nu_ext, best_rate


def calibrate_nu_ext_robust(condition, g_ratio, conn_matrix, N_per_region,
                             n_regions, coupling_strength,
                             target_rate=TARGET_RATE,
                             tolerance=TOLERANCE, calib_seeds=(42, 142, 242)):
    """Multi-seed calibration -> median nu_ext, mean achieved rate."""
    nus, rates = [], []
    for s in calib_seeds:
        nu, rate = calibrate_nu_ext(
            condition, g_ratio, conn_matrix,
            N_per_region=N_per_region, n_regions=n_regions,
            coupling_strength=coupling_strength,
            target_rate=target_rate, tolerance=tolerance, seed=s,
        )
        nus.append(nu)
        rates.append(rate)
    nu_final = float(np.median(nus))
    print(f"    -> median nu_ext across seeds {calib_seeds}: {nu_final:.4f} "
          f"(individual: {[f'{n:.4f}' for n in nus]}, "
          f"rates: {[f'{r:.2f}' for r in rates]})")
    return nu_final, float(np.mean(rates))


# ============================================================
# Common frequency grid helper
# ============================================================

def interpolate_to_common_grid(f, Pxx, f_common):
    return np.interp(f_common, f, Pxx)


# ============================================================
# Plotting — Mean ± SD
# ============================================================

def plot_mean_std(all_runs, conditions, colors,
                  band_boundaries, band_centers, band_names):
    fig, ax = plt.subplots(figsize=(12, 6))
    f_common = np.linspace(1.0, 40.0, 500)

    for condition in conditions:
        runs_Pxx = [
            interpolate_to_common_grid(run[condition]['f'], run[condition]['Pxx'], f_common)
            for run in all_runs if condition in run
        ]
        if not runs_Pxx:
            continue

        runs_array = np.array(runs_Pxx)
        mean_Pxx   = runs_array.mean(axis=0)
        std_Pxx    = runs_array.std(axis=0)

        color = colors[condition]
        ax.plot(f_common, mean_Pxx, label=condition,
                linewidth=2.5, color=color, alpha=0.9)
        ax.fill_between(f_common, mean_Pxx - std_Pxx, mean_Pxx + std_Pxx,
                        color=color, alpha=0.25)

    for boundary in band_boundaries:
        ax.axvline(x=boundary, color='gray', linestyle='--', linewidth=1.0, alpha=0.4)

    y_max = ax.get_ylim()[1]
    for center, name in zip(band_centers, band_names):
        ax.text(center, y_max * 0.95, name,
                horizontalalignment='center', fontsize=10,
                style='italic', color='gray', alpha=0.7)

    ax.set_xlabel("Frequency (Hz)", fontsize=14, fontweight='bold')
    ax.set_ylabel("Relative Power",  fontsize=14, fontweight='bold')
    ax.set_xlim([1, 40])
    ax.grid(alpha=0.3, linewidth=0.5)
    ax.legend(fontsize=11, loc='upper right')

    plt.tight_layout()
    plt.savefig("ei_fc_izhikevich_broadband_mean_std.png", dpi=300, bbox_inches='tight')
    print("Saved: ei_fc_izhikevich_broadband_mean_std.png")
    plt.close()


# ============================================================
# Plotting — Overlay of all runs
# ============================================================

def plot_overlay(all_runs, conditions, colors,
                 band_boundaries, band_centers, band_names, n_runs):
    from matplotlib.lines import Line2D

    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)
    f_common  = np.linspace(1.0, 40.0, 500)

    for ax_idx, condition in enumerate(conditions):
        ax    = axes[ax_idx]
        color = colors[condition]

        runs_Pxx = []
        for run_idx, run_data in enumerate(all_runs):
            if condition not in run_data:
                continue
            Pxx_interp = interpolate_to_common_grid(
                run_data[condition]['f'], run_data[condition]['Pxx'], f_common
            )
            runs_Pxx.append(Pxx_interp)
            ax.plot(f_common, Pxx_interp,
                    linewidth=1.2, color=color, alpha=0.35,
                    label=f"Run {run_idx+1}" if run_idx == 0 else "_nolegend_")

        if runs_Pxx:
            mean_Pxx = np.array(runs_Pxx).mean(axis=0)
            ax.plot(f_common, mean_Pxx, linewidth=3.0, color=color,
                    alpha=1.0, label="Mean", zorder=5)

        for boundary in band_boundaries:
            ax.axvline(x=boundary, color='gray', linestyle='--', linewidth=1.0, alpha=0.4)

        y_max = ax.get_ylim()[1]
        for center, name in zip(band_centers, band_names):
            ax.text(center, y_max * 0.95, name,
                    horizontalalignment='center', fontsize=10,
                    style='italic', color='gray', alpha=0.7)

        ax.set_xlabel("Frequency (Hz)", fontsize=13, fontweight='bold')
        if ax_idx == 0:
            ax.set_ylabel("Relative Power", fontsize=13, fontweight='bold')
        ax.set_xlim([1, 40])
        ax.set_title(f"{condition} – All {n_runs} Runs",
                     fontsize=13, fontweight='bold', color=color)
        ax.grid(alpha=0.3, linewidth=0.5)

        legend_elements = [
            Line2D([0], [0], color=colors[c], linewidth=2.5, label=c)
            for c in conditions
        ]
        ax.legend(handles=legend_elements, fontsize=11, loc='upper right')

    fig.suptitle("E/I Balance Spectrum (Izhikevich, Broadband) – Overlay of All Runs",
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig("ei_fc_izhikevich_broadband_overlay.png", dpi=300, bbox_inches='tight')
    print("Saved: ei_fc_izhikevich_broadband_overlay.png")
    plt.close()


# ============================================================
# Parameters
# ============================================================

DATA_ROOT   = "../ds004504/"
N_REGIONS    = 19
N_PER_REGION = 100 #21
N_RUNS       = 10

CONDITIONS = {
    'HC': 6.5,
    'AD': 2.5,
}

# === CONDITION-DEPENDENT COUPLING (new) ===
# Single-step gain nudges (G_E_RECURRENT / a shared coupling_strength)
# turned out to behave unpredictably near the oscillatory threshold --
# a small gain DECREASE (1.7->1.6) made AD's alpha peak grow instead of
# shrink, the opposite of the predicted direction. That's expected
# near a bifurcation (high sensitivity to small parameter/seed changes)
# and isn't something further single-point nudges reliably fix.
#
# More robust approach: instead of relying on HC and AD happening to
# sit at different distances from a SHARED gain threshold, make the
# alpha-driving inter-region coupling_strength itself scale with
# g_ratio. HC (strong inhibition) gets proportionally MORE coupling
# gain; AD (weak inhibition) gets proportionally LESS -- directly
# encoding "AD has both weaker inhibition AND weaker large-scale
# synchronization", which also better matches the literature (AD is
# characterized by reduced functional connectivity, not just an
# inhibition deficit).
COUPLING_STRENGTH_BASE = 1.2   # applies at the mean g_ratio across conditions

def coupling_strength_for(g_ratio, all_g_ratios, base=COUPLING_STRENGTH_BASE):
    """Scale coupling_strength proportionally to g_ratio, so a condition
    with above-average inhibition gets above-average alpha-driving
    coupling, and vice versa."""
    reference = sum(all_g_ratios) / len(all_g_ratios)
    return base * (g_ratio / reference)

COUPLING_STRENGTH_BY_CONDITION = {
    cond: coupling_strength_for(g, CONDITIONS.values())
    for cond, g in CONDITIONS.items()
}

COLORS = {
    "AD": "#90EE90",
    "HC": "#A9A9A9",
}

BAND_BOUNDARIES = [4, 8, 13, 30]
BAND_CENTERS    = [(1+4)/2, (4+8)/2, (8+13)/2, (13+30)/2, (30+40)/2]
BAND_NAMES      = ['Delta', 'Theta', 'Alpha', 'Beta', 'Gamma']

CALIB_SEEDS = (42, 142, 242)


# ============================================================
# Main execution
# ============================================================

print("=" * 70)
print("IMPROVED E/I BALANCE - FC-DRIVEN NETWORK (IZHIKEVICH, BROADBAND)")
print("=" * 70)
print(f"  - {N_RUNS} independent runs per condition")
print(f"  - Single broadband network per condition (alpha FC backbone)")
print(f"  - Spectral tilt (1/f^{SPECTRAL_TILT_EXPONENT}) applied post-hoc to each run's PSD")
print(f"  - Plot 1 : Mean +/- 1 SD")
print(f"  - Plot 2 : Overlay of all runs")
print(f"  - CLI    : 95 % bootstrapped CI per band")
print(f"  - CLI    : Cohen's d (pooled SD) AD vs HC per band")
print("=" * 70)

# === load HC and AD alpha matrices: global normalization + the single
# structural backbone used for the broadband simulation ===
print("\nLoading HC and AD alpha connectivity...")
hc_alpha = load_connectivity_matrix("HC", "alpha", DATA_ROOT)
ad_alpha = load_connectivity_matrix("AD", "alpha", DATA_ROOT)
GLOBAL_MAX = max(hc_alpha.max(), ad_alpha.max())
print(f"Global maximum FC (across HC/AD alpha): {GLOBAL_MAX:.4f}")

_, hc_region_conn = select_regions(hc_alpha, N_REGIONS)
_, ad_region_conn = select_regions(ad_alpha, N_REGIONS)
REGION_CONN = {"HC": hc_region_conn, "AD": ad_region_conn}

print("\nPer-condition coupling_strength (scaled by g_ratio):")
for cond, cs in COUPLING_STRENGTH_BY_CONDITION.items():
    print(f"  {cond:>3} (g_ratio={CONDITIONS[cond]}): coupling_strength={cs:.4f}")

# ============================================================
# Calibrate nu_ext ONCE PER CONDITION (each on its own g_ratio)
# ============================================================
print("\n" + "=" * 70)
print("Calibrating external drive (nu_ext) PER CONDITION (broadband)")
print("=" * 70)

CALIBRATED_NU_EXT = {}   # keyed by condition

for condition, g_ratio in CONDITIONS.items():
    print(f"\n--- Condition: {condition} (g_ratio={g_ratio}) ---")
    nu_final, mean_rate = calibrate_nu_ext_robust(
        condition, g_ratio, REGION_CONN[condition],
        N_per_region=N_PER_REGION, n_regions=N_REGIONS,
        coupling_strength=COUPLING_STRENGTH_BY_CONDITION[condition],
        target_rate=TARGET_RATE, tolerance=TOLERANCE,
        calib_seeds=CALIB_SEEDS,
    )
    CALIBRATED_NU_EXT[condition] = nu_final
    print(f"  {condition}: nu_ext={nu_final:.4f} (mean achieved rate ~{mean_rate:.2f} Hz)")

print("\nCalibrated nu_ext (per condition):")
for cond, nu in CALIBRATED_NU_EXT.items():
    print(f"  {cond:>3}: nu_ext={nu:.4f}")

# ============================================================
# Main execution — repeated runs
# ============================================================

print("\n" + "=" * 70)
print("Running main simulations")
print("=" * 70)

all_runs = []   # list of {condition: {f, Pxx}} dicts
rate_log = {c: [] for c in CONDITIONS}

for run_idx in range(N_RUNS):
    run_seed = 42 + run_idx * 100

    print(f"\n{'='*70}")
    print(f"RUN {run_idx + 1} / {N_RUNS}  (seed offset = {run_seed})")
    print(f"{'='*70}")

    run_results = {}

    for condition, g_ratio in CONDITIONS.items():
        print(f"\n  CONDITION: {condition}")
        try:
            result = run_simulation_with_fc(
                condition=condition,
                g_ratio=g_ratio,
                conn_matrix=REGION_CONN[condition],
                N_per_region=N_PER_REGION,
                n_regions=N_REGIONS,
                coupling_strength=COUPLING_STRENGTH_BY_CONDITION[condition],
                nu_ext=CALIBRATED_NU_EXT[condition],
                sim_time=10000.0,
                warmup=3000.0,
                seed=run_seed,
                record_fraction=0.2,
                smooth_spectrum=True,
            )
            if result.get('success', False):
                run_results[condition] = {'f': result['f'], 'Pxx': result['Pxx']}
                print(f"      OK spike rate: {result['spike_rate']:.2f} Hz")
                rate_log[condition].append(result['spike_rate'])
            else:
                print(f"      FAILED")
        except Exception as e:
            print(f"      ERROR: {e}")

    all_runs.append(run_results)

    for condition, data in run_results.items():
        alpha_mask = (data['f'] >= 8) & (data['f'] <= 13)
        if np.any(alpha_mask):
            peak_freq  = data['f'][alpha_mask][np.argmax(data['Pxx'][alpha_mask])]
            peak_power = np.max(data['Pxx'][alpha_mask])
            print(f"    {condition} alpha peak: {peak_freq:.1f} Hz "
                  f"(power: {peak_power:.4f})")


# ============================================================
# Spike-rate diagnostic
# ============================================================
print("\n" + "=" * 70)
print("SPIKE-RATE DIAGNOSTIC: AD vs HC, across all runs")
print("=" * 70)
print(f"  {'Condition':<10} {'n runs':>8}  {'Mean Hz':>10}  {'Std Hz':>10}")
print(f"  {'-'*42}")
for condition in CONDITIONS:
    rates = rate_log[condition]
    if rates:
        r = np.array(rates)
        print(f"  {condition:<10} {len(r):>8}  "
              f"{r.mean():>10.3f}  {r.std(ddof=1) if len(r) > 1 else 0.0:>10.3f}")
    else:
        print(f"  {condition:<10} {'0':>8}  {'--':>10}  {'--':>10}")

hc_rates = rate_log.get('HC', [])
ad_rates = rate_log.get('AD', [])
if hc_rates and ad_rates:
    hc_mean, ad_mean = np.mean(hc_rates), np.mean(ad_rates)
    print(f"\n  Overall mean rate — HC: {hc_mean:.2f} Hz   AD: {ad_mean:.2f} Hz")
    if ad_mean > 3 * hc_mean:
        print("  WARNING: AD rate is >3x HC — network likely saturated/near-Poisson.")
    elif ad_mean < hc_mean / 3:
        print("  WARNING: AD rate is <1/3 HC — network likely under-driven/near-silent.")
elif hc_rates and not ad_rates:
    print("\n  WARNING: No successful AD runs recorded.")
elif ad_rates and not hc_rates:
    print("\n  WARNING: No successful HC runs recorded.")


# ============================================================
# Aggregate spectra onto a common frequency grid
# ============================================================

f_common = np.linspace(1.0, 40.0, 500)

powers_by_condition = {}
n_ok_by_condition   = {}

for condition in CONDITIONS:
    runs_Pxx = [
        interpolate_to_common_grid(run[condition]['f'], run[condition]['Pxx'], f_common)
        for run in all_runs if condition in run
    ]
    if not runs_Pxx:
        print(f"\n  No successful runs for {condition} — skipping statistics.")
        continue
    powers_by_condition[condition] = np.array(runs_Pxx)
    n_ok_by_condition[condition]   = len(runs_Pxx)
    print(f"\n  {condition}: {len(runs_Pxx)}/{N_RUNS} successful runs aggregated")


stats = {}

for condition, arr in powers_by_condition.items():
    n_ok = n_ok_by_condition[condition]

    mean_power = arr.mean(axis=0)
    std_power  = arr.std(axis=0, ddof=1)
    ci_low, ci_high = bootstrap_ci(
        arr, n_boot=2000, alpha=0.05,
        rng_seed=int(10 * CONDITIONS[condition])
    )
    stats[condition] = {
        "mean":    mean_power,
        "std":     std_power,
        "ci_low":  ci_low,
        "ci_high": ci_high,
        "n":       n_ok,
    }
    print(f"  {condition}: 95 % CI computed (bootstrap, n_boot=2000)")


# ============================================================
# Print 95 % bootstrapped CI per condition and band
# ============================================================

print("\n" + "=" * 60)
print("95 % Bootstrapped Confidence Intervals (per band)")
print("=" * 60)

for condition, s in stats.items():
    print(f"\n  Condition : {condition}  "
          f"(g_I/g_E = {CONDITIONS[condition]},  n = {s['n']})")
    print(f"  {'Band':<10} {'Freq (Hz)':<14} {'Mean power':>12}  "
          f"{'CI low':>12}  {'CI high':>12}")
    print(f"  {'-'*62}")
    for bname, flo, fhi in BANDS:
        mask = (f_common >= flo) & (f_common < fhi)
        if mask.any():
            mean_b = s['mean'][mask].mean()
            ci_lo  = s['ci_low'][mask].mean()
            ci_hi  = s['ci_high'][mask].mean()
            print(f"  {bname:<10} {flo:>2}–{fhi:<5} Hz    "
                  f"{mean_b:>12.6f}  {ci_lo:>12.6f}  {ci_hi:>12.6f}")


# ============================================================
# Cohen's d (pooled SD) — AD vs HC
# ============================================================

if 'AD' in stats and 'HC' in stats:
    ad = stats['AD']
    hc = stats['HC']

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
          f"at {f_common[peak_idx]:.1f} Hz")

    print(f"\n  {'Band':<10} {'Freq (Hz)':<14} {'mean_AD':>10}  {'mean_HC':>10}  "
          f"{'s_pooled':>10}  {'Cohen d':>10}  {'|d|':>8}")
    print(f"  {'-'*80}")

    for bname, flo, fhi in BANDS:
        mask  = (f_common >= flo) & (f_common < fhi)
        if mask.any():
            m_ad  = ad['mean'][mask].mean()
            m_hc  = hc['mean'][mask].mean()
            sp    = s_pooled[mask].mean()
            d_val = d_spectrum[mask].mean()
            print(f"  {bname:<10} {flo:>2}–{fhi:<5} Hz    "
                  f"{m_ad:>10.6f}  {m_hc:>10.6f}  {sp:>10.6f}  "
                  f"{d_val:>10.4f}  {abs(d_val):>8.4f}")
else:
    print("\n  Skipping Cohen's d (need both AD and HC results)")


# ============================================================
# Plot 1: Mean ± SD
# ============================================================
print(f"\n{'='*70}")
print("PLOTTING: Mean ± SD")
print(f"{'='*70}")
plot_mean_std(all_runs, list(CONDITIONS.keys()), COLORS,
              BAND_BOUNDARIES, BAND_CENTERS, BAND_NAMES)

# ============================================================
# Plot 2: Overlay
# ============================================================
print(f"\n{'='*70}")
print("PLOTTING: Overlay of all runs")
print(f"{'='*70}")
plot_overlay(all_runs, list(CONDITIONS.keys()), COLORS,
             BAND_BOUNDARIES, BAND_CENTERS, BAND_NAMES, N_RUNS)

print("\n" + "=" * 70)
print("COMPLETE – Outputs:")
print("  • ei_fc_izhikevich_broadband_mean_std.png")
print("  • ei_fc_izhikevich_broadband_overlay.png")
print("=" * 70)
