import ROOT
import numpy as np
import yaml
import matplotlib.pyplot as plt
from analysis_dataloader import AnalysisDataloader
import awkward as ak
import mplhep as hep
import pandas as pd

hep.style.use(hep.style.CMS)
from cycler import cycler

plt.rcParams["axes.prop_cycle"] = cycler(
    "color",
    [
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
        "#e377c2",
        "#7f7f7f",
        "#bcbd22",
        "#17becf",
    ],
)


def DivideWithErrors(numerator, numerator_error, dividend, dividend_error):
    quotient = numerator / dividend
    error = quotient * np.sqrt(
        (numerator_error / numerator) ** 2 + (dividend_error / dividend) ** 2
    )
    return quotient, error


def np_to_TVector(array):
    vector = ROOT.TVector(len(array))
    for i, entry in enumerate(array):
        vector[i] = entry
    return vector


def np_to_TVectorD(array):
    vector = ROOT.TVectorD(len(array))
    for i, entry in enumerate(array):
        vector[i] = entry
    return vector


def TVector_to_np(vector):
    out_array = []
    for i in range(vector.GetNoElements()):
        out_array.append(vector[i])
    return np.array(out_array)


def plot_unfolded(
    sim_vals,
    data_vals,
    unfolding_weights,
    sim_label,
    data_label,
    unfolded_label,
    bins,
    binning_range,
    xlabel,
    title,
    outfile,
    ylabel="Normalized Entries",
    title_position=0.95,
    sim_weights=None,
    data_weights=None,
):
    if sim_weights is None:
        sim_weights = np.ones(len(sim_vals))
    if data_weights is None:
        data_weights = np.ones(len(data_vals))

    counts_sim, edges = np.histogram(
        sim_vals, bins=bins, range=binning_range, weights=sim_weights
    )
    counts_data, _ = np.histogram(
        data_vals, bins=bins, range=binning_range, weights=data_weights
    )
    counts_unf, _ = np.histogram(
        sim_vals,
        bins=bins,
        range=binning_range,
        weights=unfolding_weights * sim_weights,
    )

    centers = 0.5 * (edges[1:] + edges[:-1])
    width = np.diff(edges)

    fig, axs = plt.subplots(
        2,
        1,
        figsize=(10, 10),
        sharex=True,
        gridspec_kw={
            "height_ratios": [2, 1],  # top panel bigger
            "hspace": 0.05,  # reduce space between panels
        },
    )
    axs = axs.flatten()

    axs[0].hist(
        sim_vals,
        weights=sim_weights,
        bins=bins,
        range=binning_range,
        density=True,
        label=sim_label,
        histtype="step",
        color="#2ca02c",
        linewidth=2,
    )

    axs[0].hist(
        data_vals,
        weights=data_weights,
        bins=bins,
        range=binning_range,
        density=True,
        label=data_label,
        histtype="step",
        color="#ff7f0e",
        linewidth=2,
    )

    norm = lambda value, sum_variable: value / (np.sum(sum_variable) * width)

    unfolded_errors = np.sqrt(counts_unf)
    normalized_unfolded_counts = norm(counts_unf, counts_unf)
    normalized_unfolded_errors = norm(unfolded_errors, counts_unf)

    nonzero_unfolded_mask = normalized_unfolded_counts > 0
    axs[0].errorbar(
        centers[nonzero_unfolded_mask],
        normalized_unfolded_counts[nonzero_unfolded_mask],
        yerr=normalized_unfolded_errors[nonzero_unfolded_mask],
        fmt="o",
        label=unfolded_label,
        color="#1f77b4",
        markersize=7,
    )

    axs[0].set_ylabel(ylabel)
    axs[0].legend(loc="upper right")

    normalized_sim_counts = norm(counts_sim, counts_sim)
    normalized_sim_errors = norm(np.sqrt(counts_sim), counts_sim)
    normalized_data_counts = norm(counts_data, counts_data)
    normalized_data_errors = norm(np.sqrt(counts_data), counts_data)

    sim_truth_ratio, sim_truth_ratio_err = DivideWithErrors(
        normalized_sim_counts[nonzero_unfolded_mask],
        normalized_sim_errors[nonzero_unfolded_mask],
        normalized_unfolded_counts[nonzero_unfolded_mask],
        normalized_unfolded_errors[nonzero_unfolded_mask],
    )
    data_truth_ratio, data_truth_ratio_err = DivideWithErrors(
        normalized_data_counts[nonzero_unfolded_mask],
        normalized_data_errors[nonzero_unfolded_mask],
        normalized_unfolded_counts[nonzero_unfolded_mask],
        normalized_unfolded_errors[nonzero_unfolded_mask],
    )

    axs[1].errorbar(
        centers[nonzero_unfolded_mask],
        sim_truth_ratio,
        yerr=sim_truth_ratio_err,
        fmt="o",
        color="#2ca02c",
        markersize=7,
    )

    axs[1].errorbar(
        centers[nonzero_unfolded_mask],
        data_truth_ratio,
        yerr=data_truth_ratio_err,
        fmt="o",
        color="#ff7f0e",
        markersize=7,
    )
    axs[1].axhline(1.0, color="red", linestyle="--", linewidth=1.5)
    axs[1].set_ylabel("Counts/Unfolded")
    axs[1].set_xlabel(xlabel)
    axs[1].set_ylim(0.5, 1.5)

    plt.tight_layout()

    plt.suptitle(title, y=title_position)
    plt.savefig(outfile)
    plt.close()


def unfolding_procedure(
    flags,
    simulation_dataloader,
    data_dataloader,
    variables_to_unfold,
    new_model_name=None,
):

    if not flags.load_omnifold_model:
        print("Setting up training data dictionaries")
        simulation_training = simulation_dataloader.get_training_data()
        data_training = data_dataloader.get_training_data()

        sim_MCreco_dict_train, sim_MCgen_dict_train, data_dict_train = {}, {}, {}
        for variable in variables_to_unfold:
            sim_MCreco_dict_train[variable] = np.array(simulation_training[0][variable])
            sim_MCgen_dict_train[variable] = np.array(
                simulation_training[1][f"MC_{variable}"]
            )
            data_dict_train[variable] = np.array(data_training[0][variable])
        df_MCgen_train = ROOT.RDF.FromNumpy(sim_MCgen_dict_train)
        df_MCreco_train = ROOT.RDF.FromNumpy(sim_MCreco_dict_train)
        df_measured_train = ROOT.RDF.FromNumpy(data_dict_train)
        sim_pass_reco_vector_train = np_to_TVector(simulation_training[2])
        data_pass_reco_vector_train = np_to_TVector(data_training[2])

        if "weights" in simulation_training[0].fields:
            sim_weights = simulation_training[0]["weights"]
        else:
            sim_weights = np.ones(len(simulation_training[0]))
        sim_weights = np_to_TVectorD(sim_weights)

        if "weights" in data_training[0].fields:
            data_weights = data_training[0]["weights"]
        else:
            data_weights = np.ones(len(data_training[0]))
        data_weights = np_to_TVectorD(data_weights)

        print("Training omnifold model")
        unbinned_unfolding = ROOT.RooUnfoldOmnifold()
        unbinned_unfolding.SetSaveDirectory("./")
        model_name = (
            new_model_name if new_model_name is not None else "clasdis_gibuu_closure"
        )
        unbinned_unfolding.SetModelSaveName(model_name)
        unbinned_unfolding.SetMCgenDataFrame(df_MCgen_train)
        unbinned_unfolding.SetMCrecoDataFrame(df_MCreco_train)
        unbinned_unfolding.SetMCPassReco(sim_pass_reco_vector_train)
        unbinned_unfolding.SetMeasuredDataFrame(df_measured_train)
        unbinned_unfolding.SetMeasuredPassReco(data_pass_reco_vector_train)
        unbinned_unfolding.SetNumIterations(flags.num_iterations)
        unbinned_unfolding.SetMCgenWeights(sim_weights)
        unbinned_unfolding.SetMCrecoWeights(sim_weights)
        unbinned_unfolding.SetMeasuredWeights(data_weights)
        _ = unbinned_unfolding.UnbinnedOmnifold()

    simulation_testing = simulation_dataloader.get_testing_data()
    data_testing = data_dataloader.get_testing_data()

    sim_MCreco_dict_test, sim_MCgen_dict_test, data_dict_test = {}, {}, {}
    for variable in variables_to_unfold:
        sim_MCreco_dict_test[variable] = np.array(simulation_testing[0][variable])
        sim_MCgen_dict_test[variable] = np.array(
            simulation_testing[1][f"MC_{variable}"]
        )
        data_dict_test[variable] = np.array(data_testing[0][variable])
    df_MCgen_test = ROOT.RDF.FromNumpy(sim_MCgen_dict_test)
    df_MCreco_test = ROOT.RDF.FromNumpy(sim_MCreco_dict_test)
    df_measured_test = ROOT.RDF.FromNumpy(data_dict_test)
    sim_pass_reco_vector_test = np_to_TVector(simulation_testing[2])
    data_pass_reco_vector_test = np_to_TVector(data_testing[2])

    new_model_name = (
        new_model_name if new_model_name is not None else "clasdis_gibuu_closure"
    )
    model_name = new_model_name if flags.model_path is None else flags.model_path

    unbinned_unfolding = ROOT.RooUnfoldOmnifold()
    unbinned_unfolding.SetTestMCgenDataFrame(df_MCgen_test)
    unbinned_unfolding.SetTestMCrecoDataFrame(df_MCreco_test)
    unbinned_unfolding.SetTestMCPassReco(sim_pass_reco_vector_test)
    unbinned_unfolding.SetLoadModelPath(f"{model_name}_iteration_0.pkl")
    test_unbinned_results = unbinned_unfolding.TestUnbinnedOmnifold()
    step1_weights = TVector_to_np(ROOT.std.get[0](test_unbinned_results))
    unbinned_unfolding.SetTestMCgenDataFrame(df_MCgen_test)
    unbinned_unfolding.SetTestMCrecoDataFrame(df_MCreco_test)
    unbinned_unfolding.SetTestMCPassReco(sim_pass_reco_vector_test)
    unbinned_unfolding.SetLoadModelPath(
        f"{model_name}_iteration_{flags.num_iterations-1}.pkl"
    )
    test_unbinned_results = unbinned_unfolding.TestUnbinnedOmnifold()
    step2_weights = TVector_to_np(ROOT.std.get[1](test_unbinned_results))

    return step1_weights, step2_weights


def _load_efficiency_maps(efficiency_file, x_binning, Q2_binning, round_decimals=6):
    """Loads the efficiency CSV and reshapes both 'efficiency' and
    'efficiency_error' into 2D arrays aligned with (x_binning, Q2_binning),
    matching histogram2d's [i=x, j=Q2] indexing. See the earlier
    _load_efficiency_map discussion for how NaN/missing rows are handled --
    the same rules apply here to both columns."""
    df_eff = pd.read_csv(efficiency_file)

    x_centers = (np.asarray(x_binning[1:]) + np.asarray(x_binning[:-1])) / 2
    Q2_centers = (np.asarray(Q2_binning[1:]) + np.asarray(Q2_binning[:-1])) / 2

    eff_lookup, err_lookup = {}, {}
    for row in df_eff.itertuples():
        key = (round(row.x, round_decimals), round(row.Q2, round_decimals))
        eff_lookup[key] = row.efficiency
        err_lookup[key] = row.efficiency_error

    efficiency_map = np.full((len(x_centers), len(Q2_centers)), np.nan)
    efficiency_error_map = np.full((len(x_centers), len(Q2_centers)), np.nan)
    missing = []
    for i, x_c in enumerate(x_centers):
        for j, q2_c in enumerate(Q2_centers):
            key = (round(x_c, round_decimals), round(q2_c, round_decimals))
            if key not in eff_lookup:
                missing.append((float(x_c), float(q2_c)))
            else:
                efficiency_map[i, j] = eff_lookup[key]
                efficiency_error_map[i, j] = err_lookup[key]

    if missing:
        raise ValueError(
            f"Efficiency map '{efficiency_file}' has no entry for these (x, Q2) bin "
            f"centers -- check that it was generated with the same x_binning/Q2_binning "
            f"used here: {missing}"
        )

    return efficiency_map, efficiency_error_map


# Returns an answer in inverse nb /GeV^2
def normalize_to_absolute_cross_section(
    x_edges, y_edges, nonnormalized_hist, nonnormalized_errors, integrated_luminosity
):
    x_widths = np.diff(x_edges)
    y_widths = np.diff(y_edges)
    bin_areas = np.outer(x_widths, y_widths)
    normalized_counts = (nonnormalized_hist) / (integrated_luminosity * bin_areas)
    normalized_errors = (nonnormalized_errors) / (integrated_luminosity * bin_areas)
    return normalized_counts / (1000 * 1000), normalized_errors / (1000 * 1000)


# Calculating the double differential cross section w.r.t. x and Q2
# By default, it does the cross section for reco-level with no unfolding
def calculate_cross_sections(
    dataloader,
    target_name,
    x_binning,
    Q2_binning,
    integrated_luminosity,
    weights=None,
    use_truth=False,
    apply_radiative_corrections=False,
    radiative_corrections_df=None,
    efficiency_file=None,
    min_bin_fill=0.95,
    min_efficiency=0.01,
    report_name=None,
):
    if use_truth:
        data_to_use = dataloader.MC[dataloader.pass_truth]
        x = data_to_use["MC_x"]
        Q2 = data_to_use["MC_Q2"]
    else:
        data_to_use = dataloader.reconstructed[dataloader.pass_reco]
        target_mask = data_to_use["target"] == target_name
        x = data_to_use["x"][target_mask]
        Q2 = data_to_use["Q2"][target_mask]

    if weights is None:
        weights = np.ones(len(x))

    # Calculating the non-normalized cross sections and errors
    nonnormalized_cross_sections, _, _ = np.histogram2d(
        np.array(x),
        np.array(Q2),
        bins=(x_binning, Q2_binning),
        weights=np.array(weights),
    )
    nonnormalized_cross_sections_errors = np.sqrt(nonnormalized_cross_sections)

    if apply_radiative_corrections:
        from radiative_corrections import ApplyCorrections

        cc, rc, rc_cc, cc_err, rc_err, rc_cc_err = ApplyCorrections(
            Q2_binning,
            x_binning,
            nonnormalized_cross_sections,
            nonnormalized_cross_sections_errors,
            radiative_corrections_df,
        )
        nonnormalized_cross_sections = rc_cc
        nonnormalized_cross_sections_errors = rc_cc_err

    absolute_cross_sections, absolute_cross_sections_errors = (
        normalize_to_absolute_cross_section(
            x_binning,
            Q2_binning,
            nonnormalized_cross_sections,
            nonnormalized_cross_sections_errors,
            integrated_luminosity,
        )
    )

    if efficiency_file is not None:
        efficiency_map, efficiency_error_map = _load_efficiency_maps(
            efficiency_file, x_binning, Q2_binning
        )

        # Fold low efficiency (and its error) into NaN: epsilon at or near zero
        # has no information to invert, and dividing by it turns a handful of
        # counts into a huge, unstable correction.
        low_efficiency = efficiency_map < min_efficiency
        efficiency_map = np.where(low_efficiency, np.nan, efficiency_map)
        efficiency_error_map = np.where(low_efficiency, np.nan, efficiency_error_map)

        with np.errstate(invalid="ignore", divide="ignore"):
            corrected_cross_sections = absolute_cross_sections / efficiency_map

            # Data (Poisson) error and efficiency (binomial/MC) error are
            # independent samples, so their relative uncertainties add in
            # quadrature.
            stat_term = absolute_cross_sections_errors / efficiency_map
            eff_term = corrected_cross_sections * (
                efficiency_error_map / efficiency_map
            )
            corrected_cross_sections_errors = np.sqrt(stat_term**2 + eff_term**2)

        absolute_cross_sections = corrected_cross_sections
        absolute_cross_sections_errors = corrected_cross_sections_errors
    # NEEDS TO BE REVIEWED!!
    partial = np.zeros_like(absolute_cross_sections, dtype=bool)
    if min_bin_fill is not None:
        # Bins the kinematic boundary cuts through collect events in only part of
        # their area but are normalized by the whole rectangle, so their cross
        # section comes out suppressed by the fill fraction. Blank them rather
        # than report a value that is wrong by a known, sometimes large, factor.
        fill = kinematic_fill_fraction(x_binning, Q2_binning)
        partial = fill < min_bin_fill
        absolute_cross_sections = np.where(partial, np.nan, absolute_cross_sections)
        absolute_cross_sections_errors = np.where(
            partial, np.nan, absolute_cross_sections_errors
        )

    if report_name is not None:
        report_bin_losses(
            report_name,
            nonnormalized_cross_sections,
            partial,
            low_efficiency if efficiency_file is not None else None,
            min_bin_fill,
            min_efficiency,
        )

    return absolute_cross_sections.flatten(), absolute_cross_sections_errors.flatten()


# NEEDS TO BE REVIEWED!!
def report_bin_losses(
    name, counts, partial, low_efficiency, min_bin_fill, min_efficiency
):
    """Says why bins dropped out. Both the fill mask and the efficiency cut blank
    bins the same way, and the caller's `> 0` query then removes them together,
    so without this the two are indistinguishable in the output."""
    populated = counts > 0
    empty = ~populated

    print(f"\n{name}: bin accounting ({counts.size} total)")
    print(f"  empty (no events)                      {int(empty.sum()):>6d}")
    print(f"  populated                              {int(populated.sum()):>6d}")
    if min_bin_fill is not None:
        lost = populated & partial
        print(f"  - cut by fill < {min_bin_fill:<22g} {int(lost.sum()):>6d}")
    if low_efficiency is not None:
        lost = populated & low_efficiency & ~partial
        print(f"  - cut by efficiency < {min_efficiency:<16g} {int(lost.sum()):>6d}")
    kept = populated & ~partial
    if low_efficiency is not None:
        kept &= ~low_efficiency
    print(f"  = reported                             {int(kept.sum()):>6d}")


# Beam/target constants and the kinematic cuts applied upstream by
# selection_functions.apply_kinematic_cuts (configs/config.yaml
# ELECTRON_KINEMATIC_CUTS). These mirror tuple_maker.cpp's
# calculate_DIS_quantities -- see CLAUDE.md's note on these constants living in
# more than one place.
E_BEAM = 10.547
PROTON_MASS = 0.938
Y_MAX = 0.85
W_MIN = 2.0
Q2_MIN = 1.0

# NEEDS TO BE REVIEWED!!


def kinematic_fill_fraction(x_binning, Q2_binning, num_nodes=64):
    """Fraction of each bin's area that the upstream kinematic cuts allow.

    The cuts rearrange into two curves in the x-Q2 plane:
        y < Y_MAX  ->  x > Q2 / (2 * M * E * Y_MAX)
        W > W_MIN  ->  x < Q2 / (Q2 + W_MIN^2 - M^2)
    so the allowed x width at a given Q2 is known in closed form and the bin's
    allowed area is a 1D integral over Q2, done here by Gauss-Legendre.

    A bin fully inside the band returns 1.0. A bin the boundary cuts through
    returns the fraction of it that can hold events -- which is exactly the
    factor by which normalize_to_absolute_cross_section under-reports it, since
    that divides by the full rectangle regardless."""
    x_binning = np.asarray(x_binning, dtype=float)
    Q2_binning = np.asarray(Q2_binning, dtype=float)

    x_low, x_high = x_binning[:-1], x_binning[1:]
    Q2_low, Q2_high = Q2_binning[:-1], Q2_binning[1:]

    nodes, weights = np.polynomial.legendre.leggauss(num_nodes)
    # Gauss-Legendre nodes mapped from [-1, 1] onto each Q2 bin: (nodes, n_Q2)
    Q2_nodes = (
        0.5 * (Q2_high - Q2_low)[None, :] * nodes[:, None]
        + 0.5 * (Q2_high + Q2_low)[None, :]
    )

    x_min_allowed = Q2_nodes / (2 * PROTON_MASS * E_BEAM * Y_MAX)
    x_max_allowed = Q2_nodes / (Q2_nodes + W_MIN**2 - PROTON_MASS**2)

    # (nodes, n_x, n_Q2) -> allowed x width at every node of every bin
    lower = np.maximum(x_low[None, :, None], x_min_allowed[:, None, :])
    upper = np.minimum(x_high[None, :, None], x_max_allowed[:, None, :])
    widths = np.clip(upper - lower, 0, None)

    allowed_area = (
        0.5 * (Q2_high - Q2_low)[None, :] * np.einsum("n,nij->ij", weights, widths)
    )
    bin_areas = (x_high - x_low)[:, None] * (Q2_high - Q2_low)[None, :]

    fill = allowed_area / bin_areas
    # Q2 > Q2_MIN is a plain bound on the Q2 axis, so it just zeroes whole rows.
    fill[:, Q2_high <= Q2_MIN] = 0.0
    return fill


def coarse_Q2_grouping(Q2_binning, num_groups=10):
    """Coarser Q2 edges for plot_cross_sections' legend, derived from the actual
    binning. Spanning the real edges (rather than a hardcoded range) is what keeps
    the highest Q2 bins from being silently dropped by the query in the plot."""
    indices = np.linspace(0, len(Q2_binning) - 1, num_groups + 1).astype(int)
    return np.asarray(Q2_binning)[np.unique(indices)]


def plot_cross_sections(
    cross_section_df,
    x_binning,
    Q2_binning,
    cross_section_name="absolute_cross_sections",
    plot_title=None,
    save_path=None,
):
    fig = plt.figure(figsize=(10, 8))

    for i, Q2_lower_edge in enumerate(Q2_binning):
        if i == len(Q2_binning) - 1:
            continue
        Q2_range = (Q2_lower_edge, Q2_binning[i + 1])

        masked_dataframe = cross_section_df.query(
            f"Q2_bin_center>{Q2_range[0]} & Q2_bin_center<{Q2_range[1]} and {cross_section_name}>0"
        )

        # The x binning is not necessarily uniform, so each point gets the width of
        # the bin it actually falls in rather than a single shared width.
        x_widths = np.diff(x_binning)
        bin_indices = np.clip(
            np.searchsorted(x_binning, masked_dataframe["x_bin_center"]) - 1,
            0,
            len(x_widths) - 1,
        )

        plot = plt.errorbar(
            masked_dataframe["x_bin_center"],
            masked_dataframe[cross_section_name],
            xerr=x_widths[bin_indices] / 2,
            yerr=masked_dataframe[f"{cross_section_name}_errors"],
            label=f"{round(Q2_range[0],2)} < $Q^2$ <{round(Q2_range[1],2)}",
            fmt="o",
        )

    plt.xlabel("x")
    plt.ylabel(f"$d \sigma/dxdQ^2~ (nb/GeV^{2})$")
    plt.legend(ncol=2, loc="upper right", fontsize=16)
    plt.tight_layout()
    plt.ylim(5 * 10**-3, 5 * 10**2)
    plt.xlim(x_binning[0], x_binning[-1])
    plt.yscale("log")

    if plot_title is not None:
        plt.title(plot_title)
    if save_path is not None:
        plt.savefig(save_path)
