# Script version of analysis/notebooks/RGE_yadism_plots.ipynb
# Compares the measured RGE cross sections (from cross_sections.py) to yadism predictions.
# Makes four sets of plots, one panel per Q2 bin:
#   1. solid-target cross section vs x, data and yadism
#   2. deuterium cross section vs x, data and yadism
#   3. data/yadism ratio for the solid target
#   4. solid/deuterium ratio, data and yadism


# THE BINNING STUFF NEEDS TO BE REVIEWED!
import argparse
import json
import os

import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np
import pandas as pd

hep.style.use(hep.style.CMS)

DEFAULT_BINNING_FILE = os.path.join(os.path.dirname(__file__), "xQ2_binning.json")

YADISM_SOLID_FILES = {
    "C": "/home/rmilton/work_dir/rge_datasets/C_yadsismpredictions.csv",
    "Cu": "/home/rmilton/work_dir/rge_datasets/Cu_yadsismpredictions.csv",
    "Pb": "/home/rmilton/work_dir/rge_datasets/Pb_yadsismpredictions.csv",
    "Al": "/home/rmilton/work_dir/rge_datasets/Al_yadsismpredictions.csv",
    "Sn": "/home/rmilton/work_dir/rge_datasets/Sn_yadsismpredictions.csv",
}
YADISM_DEUTERIUM_FILE = "/home/rmilton/work_dir/rge_datasets/LD2_yadsismpredictions.csv"

# yadism cross sections are in pb/GeV^2, the RGE ones in nb/GeV^2
PB_TO_NB = 1000.0

NCOLS = 5

# Per-panel figure size, so the grid scales with the number of Q2 bins.
PANEL_WIDTH = 9.6
PANEL_HEIGHT = 5.8


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run_number",
        default=None,
        help="Run number. Only labels the plots and output file names",
        type=int,
    )
    parser.add_argument("--target", default="C", type=str)
    parser.add_argument(
        "--solid_file",
        required=True,
        help="RGE solid-target cross section .csv, from cross_sections.py",
        type=str,
    )
    parser.add_argument(
        "--deuterium_file",
        required=True,
        help="RGE deuterium cross section .csv, from cross_sections.py",
        type=str,
    )
    parser.add_argument(
        "--yadism_solid_file",
        default=None,
        help="Yadism prediction .csv for the solid target. Defaults to the target's entry",
        type=str,
    )
    parser.add_argument(
        "--yadism_deuterium_file",
        default=YADISM_DEUTERIUM_FILE,
        type=str,
    )
    parser.add_argument(
        "--cross_section_name",
        default="cross_section_withrad_nounfolding",
        help="Column of the RGE .csv files to plot",
        type=str,
    )
    parser.add_argument(
        "--ratio_cross_section_name",
        default="cross_section_norad_nounfolding",
        help="Column used for the solid/deuterium ratio plot",
        type=str,
    )
    parser.add_argument(
        "--max_relative_error",
        default=None,
        help="Drop points whose relative error exceeds this (e.g. 0.3). Off by default",
        type=float,
    )
    parser.add_argument(
        "--binning_file",
        default=DEFAULT_BINNING_FILE,
        help="Binning .json written by derive_xQ2_binning.py. Sets the Q2 bins that get a panel, the panel grid size, and the x axis range",
        type=str,
    )
    parser.add_argument("--output_directory", default="./", type=str)
    return parser.parse_args()


def read_binning(binning_file):
    """Bin centers from the derived binning. The Q2 centers are rounded the same way
    read_RGE_csv rounds its Q2 column, so the np.isclose match in merge_in_Q2_bin
    still works against non-uniform edges."""
    with open(binning_file) as file:
        binning = json.load(file)

    x_edges = np.asarray(binning["x_edges"])
    Q2_edges = np.asarray(binning["Q2_edges"])
    Q2_bin_centers = np.round((Q2_edges[1:] + Q2_edges[:-1]) / 2, 3)

    return x_edges, Q2_bin_centers


def read_RGE_csv(file_path):
    df = pd.read_csv(file_path)
    df = df.rename(columns={"x_bin_center": "x", "Q2_bin_center": "Q2"})
    df["Q2"] = np.round(df["Q2"], 3)
    return df


def read_yadism_csv(file_path):
    df = pd.read_csv(file_path, sep=",")
    df = df.rename(
        columns={
            "sigma_yadism_pb_per_GeV2": "sigma_yadism",
            "sigma_yadism_pdf_err68": "sigma_yadism_err",
        }
    )
    df["Q2"] = np.round(df["Q2"], 3)
    df["sigma_yadism"] /= PB_TO_NB
    df["sigma_yadism_err"] /= PB_TO_NB
    return df


def select_in_Q2_bin(df, Q2_bin_center):
    """The rows of one dataframe inside a single Q2 bin, sorted by x."""
    return df[np.isclose(df["Q2"], Q2_bin_center)].sort_values("x")


def merge_in_Q2_bin(left_df, right_df, Q2_bin_center, suffixes=("_x", "_y")):
    """Merges the two dataframes on x, within a single Q2 bin. The merge is on
    exact x values, so it comes back empty whenever the two sides were evaluated
    on different x grids -- which is why the plots below draw the RGE data from
    select_in_Q2_bin and only use this for the yadism overlay."""
    left_in_bin = select_in_Q2_bin(left_df, Q2_bin_center)
    right_in_bin = select_in_Q2_bin(right_df, Q2_bin_center)
    merged = left_in_bin.merge(right_in_bin, on="x", suffixes=suffixes)
    return merged.sort_values("x")


def make_panels(num_panels):
    nrows = int(np.ceil(num_panels / NCOLS))
    fig, axs = plt.subplots(
        figsize=(PANEL_WIDTH * NCOLS, PANEL_HEIGHT * nrows), ncols=NCOLS, nrows=nrows
    )
    fig.subplots_adjust(hspace=0.6)
    return fig, axs.flatten()


def format_panel(ax, Q2_bin_center, y_label, x_limits, y_limits=None):
    ax.set_xlim(*x_limits)
    ax.set_title(f"$Q^2 = {round(Q2_bin_center, 3)} ~GeV^2$", fontsize=24)
    ax.set_xlabel("x", fontsize=24)
    ax.set_ylabel(y_label, fontsize=24)
    ax.legend(fontsize=24)
    ax.grid()
    if y_limits is not None:
        ax.set_ylim(*y_limits)


def warn_if_no_yadism(panels_with_yadism, output_path):
    if panels_with_yadism == 0:
        print(
            f"  WARNING: no yadism points matched the data's x values in any Q2 bin,"
            f" so {os.path.basename(output_path)} has no prediction drawn."
            f" Regenerate the yadism predictions on the current binning."
        )


def report_grid_overlap(RGE_df, yadism_df, label):
    """Says how much the two x grids actually have in common. The merge is exact,
    so this is the thing to look at when a prediction fails to show up."""
    RGE_x = np.unique(RGE_df["x"])
    yadism_x = np.unique(yadism_df["x"])
    shared = np.intersect1d(RGE_x, yadism_x)
    print(
        f"{label}: {len(RGE_x)} RGE x values, {len(yadism_x)} yadism x values,"
        f" {len(shared)} shared"
    )
    if len(shared) == 0:
        print(
            f"  the two are on different x grids"
            f" (RGE {RGE_x.min():.4f}..{RGE_x.max():.4f},"
            f" yadism {yadism_x.min():.4f}..{yadism_x.max():.4f});"
            f" data will be drawn without a prediction"
        )


def save_figure(fig, title, output_path):
    fig.tight_layout()
    fig.suptitle(title, y=1.01, fontsize=48)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    print("Saved", output_path)


def plot_cross_sections(
    RGE_df,
    yadism_df,
    Q2_bin_centers,
    x_limits,
    cross_section_name,
    max_relative_error,
    title,
    output_path,
):
    """Cross section vs x per Q2 bin, RGE data against the yadism prediction."""
    error_name = cross_section_name + "_errors"
    fig, axs = make_panels(len(Q2_bin_centers))

    panels_with_yadism = 0

    for i, Q2_bin_center in enumerate(Q2_bin_centers):
        RGE_in_bin = select_in_Q2_bin(RGE_df, Q2_bin_center)

        if max_relative_error is not None:
            relative_error = RGE_in_bin[error_name] / RGE_in_bin[cross_section_name]
            RGE_in_bin = RGE_in_bin[
                np.isfinite(relative_error)
                & (relative_error.abs() < max_relative_error)
            ]

        if RGE_in_bin.empty:
            continue

        axs[i].errorbar(
            RGE_in_bin["x"],
            RGE_in_bin[cross_section_name],
            yerr=RGE_in_bin[error_name],
            fmt="o",
            label="Reco RGE data",
            markersize=12,
        )

        # The yadism prediction is only drawn where it was evaluated at the same x
        # points as the data. If it was computed on a different binning there is
        # nothing to overlay, but the data above is still plotted.
        merged = merge_in_Q2_bin(RGE_in_bin, yadism_df, Q2_bin_center)
        if not merged.empty:
            panels_with_yadism += 1
            sigma_yadism = merged["sigma_yadism"]
            sigma_yadism_err = merged["sigma_yadism_err"]
            axs[i].plot(merged["x"], sigma_yadism, "s", label="Yadism", markersize=12)
            axs[i].fill_between(
                merged["x"],
                sigma_yadism - sigma_yadism_err,
                sigma_yadism + sigma_yadism_err,
                alpha=0.3,
            )

        format_panel(
            axs[i],
            Q2_bin_center,
            r"$d^2 \sigma / (dQ^2 dx)~ (nb/GeV^2)$",
            x_limits,
        )

    warn_if_no_yadism(panels_with_yadism, output_path)
    save_figure(fig, title, output_path)


def plot_data_over_yadism(
    RGE_df,
    yadism_df,
    Q2_bin_centers,
    x_limits,
    cross_section_name,
    title,
    output_path,
):
    """Ratio of the measured cross section to the yadism prediction, per Q2 bin."""
    error_name = cross_section_name + "_errors"
    fig, axs = make_panels(len(Q2_bin_centers))
    panels_with_yadism = 0

    for i, Q2_bin_center in enumerate(Q2_bin_centers):
        # This plot is a ratio to yadism, so unlike the others there is nothing to
        # draw when the prediction has no point at the data's x values.
        merged = merge_in_Q2_bin(RGE_df, yadism_df, Q2_bin_center)
        if merged.empty:
            continue
        panels_with_yadism += 1

        sigma_data = merged[cross_section_name]
        sigma_data_err = merged[error_name]
        sigma_yadism = merged["sigma_yadism"]
        sigma_yadism_err = merged["sigma_yadism_err"]

        ratio = sigma_data / sigma_yadism
        ratio_err = ratio * np.sqrt(
            (sigma_data_err / sigma_data) ** 2 + (sigma_yadism_err / sigma_yadism) ** 2
        )

        axs[i].errorbar(
            merged["x"], ratio, yerr=ratio_err, fmt="o", label="Reco RGE data"
        )
        format_panel(axs[i], Q2_bin_center, "RGE / yadism", x_limits, y_limits=(0, 1))

    warn_if_no_yadism(panels_with_yadism, output_path)
    save_figure(fig, title, output_path)


def plot_solid_over_deuterium(
    RGE_solid_df,
    RGE_deuterium_df,
    yadism_solid_df,
    yadism_deuterium_df,
    Q2_bin_centers,
    x_limits,
    cross_section_name,
    title,
    output_path,
):
    """Solid/deuterium cross section ratio per Q2 bin, data against yadism."""
    fig, axs = make_panels(len(Q2_bin_centers))

    for i, Q2_bin_center in enumerate(Q2_bin_centers):
        merged = merge_in_Q2_bin(
            RGE_solid_df,
            RGE_deuterium_df,
            Q2_bin_center,
            suffixes=("_solid", "_deuterium"),
        )
        yadism_merged = merge_in_Q2_bin(
            yadism_solid_df,
            yadism_deuterium_df,
            Q2_bin_center,
            suffixes=("_solid", "_deuterium"),
        )
        if merged.empty:
            continue

        axs[i].errorbar(
            merged["x"],
            merged[cross_section_name + "_solid"]
            / merged[cross_section_name + "_deuterium"],
            fmt="o",
            label="Reco RGE data",
        )

        # Only show the prediction where there is data to compare it to
        yadism_merged = yadism_merged[yadism_merged["x"].isin(merged["x"])]
        axs[i].plot(
            yadism_merged["x"],
            yadism_merged["sigma_yadism_solid"]
            / yadism_merged["sigma_yadism_deuterium"],
            "s",
            label="Yadism",
        )
        format_panel(
            axs[i], Q2_bin_center, r"$\sigma_{solid} / \sigma_{deuterium}$", x_limits
        )

    save_figure(fig, title, output_path)


def main():
    flags = parse_arguments()
    os.makedirs(flags.output_directory, exist_ok=True)

    yadism_solid_file = flags.yadism_solid_file or YADISM_SOLID_FILES[flags.target]

    print("RGE solid target file:", flags.solid_file)
    print("RGE deuterium file:", flags.deuterium_file)
    print("Yadism solid target file:", yadism_solid_file)
    print("Yadism deuterium file:", flags.yadism_deuterium_file)

    RGE_solid_df = read_RGE_csv(flags.solid_file)
    RGE_deuterium_df = read_RGE_csv(flags.deuterium_file)
    yadism_solid_df = read_yadism_csv(yadism_solid_file)
    yadism_deuterium_df = read_yadism_csv(flags.yadism_deuterium_file)

    # The Q2 bins that get a panel come from the derived binning, not from whatever
    # grid the yadism predictions happen to be on.
    x_edges, Q2_bin_centers = read_binning(flags.binning_file)
    x_limits = (x_edges[0], x_edges[-1])
    print("Binning file:", flags.binning_file)

    report_grid_overlap(RGE_solid_df, yadism_solid_df, "Solid target")
    report_grid_overlap(RGE_deuterium_df, yadism_deuterium_df, "Deuterium")

    run_label = "RGE" if flags.run_number is None else f"RGE {flags.run_number}"
    file_prefix = run_label.replace(" ", "_")

    def output_path(name):
        return os.path.join(
            flags.output_directory,
            f"{file_prefix}_{flags.target}_{name}.png",
        )

    plot_cross_sections(
        RGE_solid_df,
        yadism_solid_df,
        Q2_bin_centers,
        x_limits,
        flags.cross_section_name,
        flags.max_relative_error,
        f"{run_label}: {flags.target} reconstructed",
        output_path("reco_crosssections"),
    )
    plot_cross_sections(
        RGE_deuterium_df,
        yadism_deuterium_df,
        Q2_bin_centers,
        x_limits,
        flags.cross_section_name,
        flags.max_relative_error,
        f"{run_label}: LD2 reconstructed",
        output_path("LD2_reco_crosssections"),
    )
    plot_data_over_yadism(
        RGE_solid_df,
        yadism_solid_df,
        Q2_bin_centers,
        x_limits,
        flags.cross_section_name,
        f"{run_label}: {flags.target} reconstructed",
        output_path("data_over_yadism"),
    )
    plot_solid_over_deuterium(
        RGE_solid_df,
        RGE_deuterium_df,
        yadism_solid_df,
        yadism_deuterium_df,
        Q2_bin_centers,
        x_limits,
        flags.ratio_cross_section_name,
        f"{run_label}: {flags.target} reconstructed",
        output_path("solid_over_deuterium"),
    )


if __name__ == "__main__":
    main()
