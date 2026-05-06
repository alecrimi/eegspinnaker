import nest
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.signal import welch
import os


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
    node_strength = np.sum(np.abs(conn_matrix), axis=1)
    selected_indices = np.argsort(node_strength)[-n_subnets:]
    subnet_conn = conn_matrix[np.ix_(selected_indices, selected_indices)]
    return selected_indices, subnet_conn


# ============================================================
# Main simulation
# ============================================================

def run_simulation_with_fc(condition_name,
                           band_name,
                           g_ratio,
                           conn_matrix,
                           N_subnet=100,        # REDUCED: 260→100 neurons per subnetwork
                           N_subnets=19,
                           frac_exc=0.8,
                           p_conn_local=0.15,
                           coupling_strength=0.5,  # REDUCED: 2.0→0.5 to avoid overdrive
                           nu_ext=2.0,             # REDUCED: 5.0→2.0 Hz (before ×1000 → mHz)
                           sim_time=5000.0,
                           warmup=1000.0,
                           seed_base=42):

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
        E_pop = nest.Create("iaf_cond_exp", N_E_subnet)
        I_pop = nest.Create("iaf_cond_exp", N_I_subnet)

        E_pop.V_m  = -70 + 5 * np.random.randn(N_E_subnet)
        I_pop.V_m  = -70 + 5 * np.random.randn(N_I_subnet)
        E_pop.V_th = -50 + 2 * np.random.randn(N_E_subnet)
        I_pop.V_th = -50 + 2 * np.random.randn(N_I_subnet)

        E_pop.t_ref = 2
        I_pop.t_ref = 1

        E_pop.tau_syn_ex = 2
        E_pop.tau_syn_in = 8
        I_pop.tau_syn_ex = 1
        I_pop.tau_syn_in = 4

        E_populations.append(E_pop)
        I_populations.append(I_pop)

    g_E_base    = 3.0
    delay_local = 1.5
    delay_inter = 3.0

    # --------------------------------------------------------
    # External drive: ONE shared Poisson generator per pop
    # (not one per neuron — that was the main desynchroniser)
    # --------------------------------------------------------
    for subnet_idx in range(N_subnets):
        E_pop = E_populations[subnet_idx]
        I_pop = I_populations[subnet_idx]

        # Shared excitatory drive — same rate for all E neurons
        pg_E = nest.Create("poisson_generator")
        pg_E.rate = nu_ext * 1000   # nu_ext in kHz → Hz
        nest.Connect(
            pg_E, E_pop,
            conn_spec={"rule": "all_to_all"},
            syn_spec={"weight": 0.5 * g_E_base, "delay": delay_local}
        )

        # Shared excitatory drive to I neurons
        pg_I = nest.Create("poisson_generator")
        pg_I.rate = nu_ext * 1000
        nest.Connect(
            pg_I, I_pop,
            conn_spec={"rule": "all_to_all"},
            syn_spec={"weight": 0.5 * g_E_base, "delay": delay_local}
        )

    # --------------------------------------------------------
    # Local recurrent connectivity
    # --------------------------------------------------------
    g_E = g_E_base
    g_I = g_ratio * g_E

    conn_spec = {"rule": "pairwise_bernoulli", "p": p_conn_local}

    for subnet_idx in range(N_subnets):
        E_pop = E_populations[subnet_idx]
        I_pop = I_populations[subnet_idx]

        nest.Connect(E_pop, E_pop, conn_spec, syn_spec={"weight": g_E,        "delay": delay_local})
        nest.Connect(E_pop, I_pop, conn_spec, syn_spec={"weight": 1.2 * g_E,  "delay": delay_local})
        nest.Connect(I_pop, E_pop, conn_spec, syn_spec={"weight": -g_I,        "delay": delay_local})
        nest.Connect(I_pop, I_pop, conn_spec, syn_spec={"weight": -0.8 * g_I,  "delay": delay_local})

    # --------------------------------------------------------
    # Inter-subnetwork FC-weighted connections
    # --------------------------------------------------------
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

    # --------------------------------------------------------
    # Recording: one multimeter per subnetwork, sample ALL E neurons
    # (we want a proper LFP proxy — MUA average)
    # --------------------------------------------------------
    multimeters = []
    for s in range(N_subnets):
        mE = nest.Create("multimeter")
        mE.set(record_from=["V_m"], interval=1.0)
        nest.Connect(mE, E_populations[s])   # record ALL E neurons, not just 20
        multimeters.append(mE)

    spike_recorders = []
    for subnet_idx in range(N_subnets):
        sr = nest.Create("spike_recorder")
        nest.Connect(E_populations[subnet_idx], sr)
        spike_recorders.append(sr)

    nest.Simulate(warmup + sim_time)

    # --------------------------------------------------------
    # Spike rates
    # --------------------------------------------------------
    spike_rates = []
    for sr in spike_recorders:
        spikes     = sr.get("events")
        spike_times = np.array(spikes["times"])
        spike_times = spike_times[spike_times > warmup]
        rate = len(spike_times) / (N_E_subnet * sim_time / 1000)
        spike_rates.append(rate)

    # --------------------------------------------------------
    # PSD: compute per-neuron, then AVERAGE PSDs
    # (NOT average V_m first — that kills oscillations!)
    # --------------------------------------------------------
    subnet_psds = []

    for mE in multimeters:
        events = mE.get("events")
        t      = np.array(events["times"])
        Vm_raw = np.array(events["V_m"])
        sender = np.array(events["senders"])

        mask = t > warmup
        t_m, Vm_m, sender_m = t[mask], Vm_raw[mask], sender[mask]

        uids = np.unique(sender_m)
        if len(uids) == 0:
            continue

        nperseg = min(2048, int(sim_time))  # 1 sample / ms → nperseg in samples

        neuron_psds = []
        for uid in uids:
            tr = Vm_m[sender_m == uid]
            if len(tr) < nperseg:
                continue
            f, Pxx = welch(tr, fs=1000.0, nperseg=nperseg, noverlap=3 * nperseg // 4)
            neuron_psds.append(Pxx)

        if not neuron_psds:
            continue

        # Average across neurons AFTER computing individual PSDs
        Pxx_mean = np.mean(neuron_psds, axis=0)
        subnet_psds.append(Pxx_mean)

    if not subnet_psds:
        return {"success": False}

    f_band = (f >= 1) & (f <= 100)
    f      = f[f_band]
    Pxx_grand = np.mean([p[f_band] for p in subnet_psds], axis=0)
    Pxx_rel   = Pxx_grand / np.sum(Pxx_grand)

    return {
        "condition":  condition_name,
        "band":       band_name,
        "f":          f,
        "Pxx":        Pxx_rel,
        "spike_rate": np.mean(spike_rates),
        "success":    True
    }


# ============================================================
# Stitch spectra
# ============================================================

def stitch_spectra(results_by_condition, band_ranges):
    stitched = {}
    for condition, band_results in results_by_condition.items():
        all_f   = []
        all_Pxx = []
        for band_name in ['delta', 'theta', 'alpha', 'beta', 'gamma']:
            if band_name not in band_results:
                continue
            result = band_results[band_name]
            f, Pxx = result['f'], result['Pxx']
            low, high = band_ranges[band_name]
            mask = (f >= low) & (f < high)
            all_f.append(f[mask])
            all_Pxx.append(Pxx[mask])

        if all_f:
            stitched_f   = np.concatenate(all_f)
            stitched_Pxx = np.concatenate(all_Pxx)
            # Re-normalise ONCE after all bands are stitched
            stitched_Pxx /= np.sum(stitched_Pxx)
            stitched[condition] = {"f": stitched_f, "Pxx": stitched_Pxx}

    return stitched


# ============================================================
# Parameters
# ============================================================

DATA_ROOT   = "../ds004504/"
BAND_RANGES = {
    "delta": (1,  4),
    "theta": (4,  8),
    "alpha": (8,  13),
    "beta":  (13, 30),
    "gamma": (30, 45),
}

# g_ratio: inhibition-to-excitation weight ratio.
# HC: moderate I/E → healthy beta/gamma balance
# AD: reduced I/E (interneuron loss) → less high-freq, more low-freq
CONDITIONS = {
    "HC": 6.5,   # was 6.5 — too strong, suppressed all oscillations
    "AD": 2.5,
}

N_SUBNETS = 19
N_SUBNET  = 120   # was 260 — computationally lighter, still enough for LFP proxy
N_REPEATS = 10     # run 3 times to get meaningful mean ± std


# ============================================================
# Repeated experiment
# ============================================================

stitched_runs = {c: [] for c in CONDITIONS}
f_axes        = {c: None for c in CONDITIONS}

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
                    condition,
                    band_name,
                    g_ratio,
                    subnet_conn,
                    N_subnet=N_SUBNET,
                    N_subnets=N_SUBNETS,
                    seed_base=42 + repeat * 100
                )
                if result["success"]:
                    results_by_condition[condition][band_name] = result
            except Exception as e:
                print("Error:", e)

    stitched = stitch_spectra(results_by_condition, BAND_RANGES)

    for condition in stitched:
        stitched_runs[condition].append(stitched[condition]["Pxx"])
        f_axes[condition] = stitched[condition]["f"]


# ============================================================
# Compute mean ± std
# ============================================================

stitched_mean_std = {}
for condition in stitched_runs:
    if not stitched_runs[condition]:
        continue
    spectra = np.vstack(stitched_runs[condition])
    stitched_mean_std[condition] = {
        "f":    f_axes[condition],
        "mean": np.mean(spectra, axis=0),
        "std":  np.std(spectra, axis=0),
    }


# ============================================================
# Plot
# ============================================================

fig, ax = plt.subplots(figsize=(10, 6))
colors = {"HC": "#A9A9A9", "AD": "#90EE90"}

for condition, data in stitched_mean_std.items():
    f, mean, std = data["f"], data["mean"], data["std"]
    #ax.semilogy(f, mean, color=colors[condition], linewidth=2.5, label=condition)
    ax.plot(f, mean, color=colors[condition], linewidth=2.5, label=condition)
    
    ax.fill_between(f, mean - std, mean + std,
                    color=colors[condition], alpha=0.25)

for b in [4, 8, 13, 30]:
    ax.axvline(b, color="gray", linestyle="--", alpha=0.5)

ax.set_xlim(1, 40)
ax.set_xlabel("Frequency (Hz)", fontsize=14, fontweight="bold")
ax.set_ylabel("Relative power", fontsize=14, fontweight="bold")
ax.legend()
ax.grid(alpha=0.3, which="both")

y_max = ax.get_ylim()[1]
band_centers = [(1+4)/2, (4+8)/2, (8+13)/2, (13+30)/2, (30+40)/2]
band_labels  = ['Delta', 'Theta', 'Alpha', 'Beta', 'Gamma']
for center, name in zip(band_centers, band_labels):
    ax.text(center, y_max * 0.7, name,
            ha='center', fontsize=9, style='italic', color='gray', alpha=0.7)

plt.tight_layout()
plt.savefig("stitched_spectrum_mean_std.png", dpi=300)
plt.savefig("stitched_spectrum_mean_std.pdf")
plt.close()
print("\nSimulation complete")
