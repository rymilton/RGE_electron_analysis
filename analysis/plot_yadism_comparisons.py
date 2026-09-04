# Script version of analysis/notebooks/RGE_yadism_plots.ipynb
# Compares the measured RGE cross sections (from cross_sections.py) to yadism predictions.
# Makes four sets of plots, one panel per Q2 bin:
#   1. solid-target cross section vs x, data and yadism
#   2. deuterium cross section vs x, data and yadism
#   3. data/yadism ratio for the solid target
#   4. solid/deuterium ratio, data and yadism

import argparse
import os

import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np
import pandas as pd

hep.style.use(hep.style.CMS)

SOLID_RUN_FILES = {
    20030: "/volatile/clas12/rmilton/rge_datasets/pass1/torus-1/C_D2/carbon_cross_sections.csv",
    20082: "/home/rmilton/work_dir/rge_datasets/phys_val/020082/Pb_cross_sections.csv",
    20226: "/home/rmilton/work_dir/rge_datasets/phys_val/020226/Cu_cross_sections.csv",
    20417: "/home/rmilton/work_dir/rge_datasets/phys_val/020417/Sn_cross_sections.csv",
    20485: "/home/rmilton/work_dir/rge_datasets/phys_val/020485/Al_cross_sections.csv",
}
DEUTERIUM_RUN_FILES = {
    20030: "/volatile/clas12/rmilton/rge_datasets/pass1/torus-1/C_D2/LD2_cross_sections.csv",
    20082: "/home/rmilton/work_dir/rge_datasets/phys_val/020082/LD2_cross_sections.csv",
    20226: "/home/rmilton/work_dir/rge_datasets/phys_val/020226/LD2_cross_sections.csv",
    20417: "/home/rmilton/work_dir/rge_datasets/phys_val/020417/LD2_cross_sections.csv",
    20485: "/home/rmilton/work_dir/rge_datasets/phys_val/020485/LD2_cross_sections.csv",
}
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
NROWS = 9


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_number", default=20030, type=int)
    parser.add_argument("--target", default="C", type=str)
    parser.add_argument(
        "--solid_file",
        default=None,
        help="RGE solid-target cross section .csv. Defaults to the run number's entry",
        type=str,
    )
    parser.add_argument(
        "--deuterium_file",
        default=None,
        help="RGE deuterium cross section .csv. Defaults to the run number's entry",
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
    parser.add_argument("--output_directory", default="./", type=str)
    return parser.parse_args()


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


def merge_in_Q2_bin(left_df, right_df, Q2_bin_center, suffixes=("_x", "_y")):
    """Merges the two dataframes on x, within a single Q2 bin."""
    left_in_bin = left_df[np.isclose(left_df["Q2"], Q2_bin_center)]
    right_in_bin = right_df[np.isclose(right_df["Q2"], Q2_bin_center)]
    merged = left_in_bin.merge(right_in_bin, on="x", suffixes=suffixes)
    return merged.sort_values("x")


def make_panels():
    fig, axs = plt.subplots(figsize=(48, 52), ncols=NCOLS, nrows=NROWS)
    fig.subplots_adjust(hspace=0.6)
    return fig, axs.flatten()


def format_panel(ax, Q2_bin_center, y_label, y_limits=None):
    ax.set_xlim(0, 1)
    ax.set_title(f"$Q^2 = {round(Q2_bin_center, 3)} ~GeV^2$", fontsize=24)
    ax.set_xlabel("x", fontsize=24)
    ax.set_ylabel(y_label, fontsize=24)
    ax.legend(fontsize=24)
    ax.grid()
    if y_limits is not None:
        ax.set_ylim(*y_limits)


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
    cross_section_name,
    max_relative_error,
    title,
    output_path,
):
    """Cross section vs x per Q2 bin, RGE data against the yadism prediction."""
    error_name = cross_section_name + "_errors"
    fig, axs = make_panels()

    for i, Q2_bin_center in enumerate(Q2_bin_centers):
        merged = merge_in_Q2_bin(RGE_df, yadism_df, Q2_bin_center)
        if merged.empty:
            continue

        if max_relative_error is not None:
            relative_error = merged[error_name] / merged[cross_section_name]
            merged = merged[
                np.isfinite(relative_error) & (relative_error.abs() < max_relative_error)
            ]
            if merged.empty:
                continue

        x = merged["x"]
        sigma_yadism = merged["sigma_yadism"]
        sigma_yadism_err = merged["sigma_yadism_err"]

        axs[i].errorbar(
            x,
            merged[cross_section_name],
            yerr=merged[error_name],
            fmt="o",
            label="Reco RGE data",
            markersize=12,
        )
        axs[i].plot(x, sigma_yadism, "s", label="Yadism", markersize=12)
        axs[i].fill_between(
            x,
            sigma_yadism - sigma_yadism_err,
            sigma_yadism + sigma_yadism_err,
            alpha=0.3,
        )
        format_panel(axs[i], Q2_bin_center, r"$d^2 \sigma / (dQ^2 dx)~ (nb/GeV^2)$")

    save_figure(fig, title, output_path)


def plot_data_over_yadism(
    RGE_df, yadism_df, Q2_bin_centers, cross_section_name, title, output_path
):
    """Ratio of the measured cross section to the yadism prediction, per Q2 bin."""
    error_name = cross_section_name + "_errors"
    fig, axs = make_panels()

    for i, Q2_bin_center in enumerate(Q2_bin_centers):
        merged = merge_in_Q2_bin(RGE_df, yadism_df, Q2_bin_center)
        if merged.empty:
            continue

        sigma_data = merged[cross_section_name]
        sigma_data_err = merged[error_name]
        sigma_yadism = merged["sigma_yadism"]
        sigma_yadism_err = merged["sigma_yadism_err"]

        ratio = sigma_data / sigma_yadism
        ratio_err = ratio * np.sqrt(
            (sigma_data_err / sigma_data) ** 2 + (sigma_yadism_err / sigma_yadism) ** 2
        )

        axs[i].errorbar(merged["x"], ratio, yerr=ratio_err, fmt="o", label="Reco RGE data")
        format_panel(axs[i], Q2_bin_center, "RGE / yadism", y_limits=(0, 1))

    save_figure(fig, title, output_path)


def plot_solid_over_deuterium(
    RGE_solid_df,
    RGE_deuterium_df,
    yadism_solid_df,
    yadism_deuterium_df,
    Q2_bin_centers,
    cross_section_name,
    title,
    output_path,
):
    """Solid/deuterium cross section ratio per Q2 bin, data against yadism."""
    fig, axs = make_panels()

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
            yadism_merged["sigma_yadism_solid"] / yadism_merged["sigma_yadism_deuterium"],
            "s",
            label="Yadism",
        )
        format_panel(axs[i], Q2_bin_center, r"$\sigma_{solid} / \sigma_{deuterium}$")

    save_figure(fig, title, output_path)


def main():
    flags = parse_arguments()
    os.makedirs(flags.output_directory, exist_ok=True)

    solid_file = flags.solid_file or SOLID_RUN_FILES[flags.run_number]
    deuterium_file = flags.deuterium_file or DEUTERIUM_RUN_FILES[flags.run_number]
    yadism_solid_file = flags.yadism_solid_file or YADISM_SOLID_FILES[flags.target]

    print("RGE solid target file:", solid_file)
    print("RGE deuterium file:", deuterium_file)
    print("Yadism solid target file:", yadism_solid_file)
    print("Yadism deuterium file:", flags.yadism_deuterium_file)

    RGE_solid_df = read_RGE_csv(solid_file)
    RGE_deuterium_df = read_RGE_csv(deuterium_file)
    yadism_solid_df = read_yadism_csv(yadism_solid_file)
    yadism_deuterium_df = read_yadism_csv(flags.yadism_deuterium_file)

    Q2_bin_centers = np.unique(yadism_solid_df["Q2"])

    def output_path(name):
        return os.path.join(
            flags.output_directory,
            f"RGE_{flags.target}_{flags.run_number}_{name}.png",
        )

    plot_cross_sections(
        RGE_solid_df,
        yadism_solid_df,
        Q2_bin_centers,
        flags.cross_section_name,
        flags.max_relative_error,
        f"RGE {flags.run_number}: {flags.target} reconstructed",
        output_path("reco_crosssections"),
    )
    plot_cross_sections(
        RGE_deuterium_df,
        yadism_deuterium_df,
        Q2_bin_centers,
        flags.cross_section_name,
        flags.max_relative_error,
        f"RGE {flags.run_number}: LD2 reconstructed",
        output_path("LD2_reco_crosssections"),
    )
    plot_data_over_yadism(
        RGE_solid_df,
        yadism_solid_df,
        Q2_bin_centers,
        flags.cross_section_name,
        f"RGE {flags.run_number}: {flags.target} reconstructed",
        output_path("data_over_yadism"),
    )
    plot_solid_over_deuterium(
        RGE_solid_df,
        RGE_deuterium_df,
        yadism_solid_df,
        yadism_deuterium_df,
        Q2_bin_centers,
        flags.ratio_cross_section_name,
        f"RGE {flags.run_number}: {flags.target} reconstructed",
        output_path("solid_over_deuterium"),
    )


if __name__ == "__main__":
    main()
