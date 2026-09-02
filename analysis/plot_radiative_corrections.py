import numpy as np
import pandas as pd
import analysis_options
import argparse
import matplotlib.pyplot as plt
from radiative_corrections import OpenCorrections
import os, sys
import mplhep as hep

hep.style.use(hep.style.CMS)
REPO_TOP_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO_TOP_DIR)
from utils import open_data

prop_cycle = plt.rcParams["axes.prop_cycle"]
colors = prop_cycle.by_key()["color"]


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--target_name",
        default="Al",
        help="Target you want the externals input for",
        type=str,
    )
    parser.add_argument(
        "--plots_directory",
        default="/home/rmilton/work_dir/rge_datasets/phys_val/020485/radiative_plots/",
        help="Directory to save plots to",
        type=str,
    )
    parser.add_argument(
        "--plot_string",
        default="with_walls",
        help="String to add to the plot file names",
        type=str,
    )
    parser.add_argument(
        "--externals_file",
        default="/home/rmilton/work_dir/externals/OUT/RGE_Al.out",
        help="Path to file produced by externals",
        type=str,
    )

    parser.add_argument(
        "--data_file",
        default=None,
        help="If desired, the path to a ROOT file so we compute the impact of cuts on the data",
        type=str,
    )

    flags = parser.parse_args()

    return flags


def make_2d_scatter_plot(
    xaxis_quantity,
    yaxis_quantity,
    zaxis_quantity,
    yaxis_quantity_name,
    yaxis_binning,
    cut=None,
    draw_original_distribution=False,
    xaxis_limits=None,
    yaxis_limits=None,
    xaxis_label=None,
    yaxis_label=None,
    title=None,
    save_path=None,
    legend_location="lower right",
    legend_ncols=1,
    legend_fontsize=18,
):
    fig = plt.figure(figsize=(12, 8))

    for i, yaxis_lower_edge in enumerate(yaxis_binning):
        if i == len(yaxis_binning) - 1:
            continue
        yaxis_range = (yaxis_lower_edge, yaxis_binning[i + 1])

        yaxis_mask = (yaxis_quantity > yaxis_range[0]) & (
            yaxis_quantity < yaxis_range[1]
        )

        if cut is not None and draw_original_distribution is True:
            color = colors[i % len(colors)]
            plt.scatter(
                xaxis_quantity[(yaxis_mask) & (~cut)],
                zaxis_quantity[(yaxis_mask) & (~cut)],
                marker="o",
                facecolors="none",
                edgecolors=color,  # Make same color as previous scatter
                label="Removed by cut" if i == 0 else "",
            )
            plt.scatter(
                xaxis_quantity[(yaxis_mask) & (cut)],
                zaxis_quantity[(yaxis_mask) & (cut)],
                label=f"{round(yaxis_range[0],2)}< {yaxis_quantity_name} <{round(yaxis_range[1],2)}",
            )

        elif cut is not None:
            plt.scatter(
                xaxis_quantity[(yaxis_mask) & (cut)],
                zaxis_quantity[(yaxis_mask) & (cut)],
                label=f"{round(yaxis_range[0],2)}< {yaxis_quantity_name} <{round(yaxis_range[1],2)}",
            )
        else:
            plt.scatter(
                xaxis_quantity[yaxis_mask],
                zaxis_quantity[yaxis_mask],
                label=f"{round(yaxis_range[0],2)}< {yaxis_quantity_name} <{round(yaxis_range[1],2)}",
            )

    plt.legend(loc=legend_location, ncols=legend_ncols, fontsize=legend_fontsize)
    plt.xlabel(xaxis_label)
    plt.ylabel(yaxis_label)
    plt.title(title)
    plt.xlim(xaxis_limits)
    plt.ylim(yaxis_limits)
    plt.savefig(save_path)


if __name__ == "__main__":
    flags = parse_arguments()
    print(f"Getting EXTERNALS input for {flags.target_name}")

    x_bins = analysis_options.x_bins_by_target[flags.target_name]
    x_bin_centers = (x_bins[1:] + x_bins[:-1]) / 2
    Q2_bins = analysis_options.Q2_bins_by_target[flags.target_name]
    Q2_bin_centers = (Q2_bins[1:] + Q2_bins[:-1]) / 2

    x_padded, Q2_padded = np.meshgrid(x_bin_centers, Q2_bin_centers)
    # Taking transpose gives x constant along inner dimension, Q2 varies along inner dimension
    x_padded = x_padded.T
    Q2_padded = Q2_padded.T

    x_bin_centers_repeated = []
    for xbins in x_padded:
        x_bin_centers_repeated.extend(xbins)
    Q2_bin_centers_repeated = []
    for Q2bins in Q2_padded:
        Q2_bin_centers_repeated.extend(Q2bins)
    Q2_bin_centers_repeated = np.array(Q2_bin_centers_repeated)
    x_bin_centers_repeated = np.array(x_bin_centers_repeated)

    beam_energy = 10.5473  # In GeV. Taken from RCDB for RGE
    proton_mass = 0.938  # In GeV

    y_bin_centers_repeated = Q2_bin_centers_repeated / (
        2 * beam_energy * proton_mass * x_bin_centers_repeated
    )

    if flags.data_file is not None:
        branches_to_open = ["Q2", "x", "y", "pass_reco"]
        data_events = open_data(
            flags.data_file,
            branches_to_open=branches_to_open,
            data_tree_name="reconstructed_electrons",
        )["reconstructed"]
        previous_y = data_events["y"]

        print(
            "The fraction of events lost by y<1 cut is ",
            1 - len(data_events[data_events["y"] < 1]) / len(data_events),
        )
        print(
            "The fraction of events lost by y<.85 cut is ",
            1 - len(data_events[data_events["y"] < 0.85]) / len(data_events),
        )
        print(
            "The fraction of events lost by y<.8 cut is ",
            1 - len(data_events[data_events["y"] < 0.8]) / len(data_events),
        )

    corrections_df = OpenCorrections(flags.externals_file)
    corrections_df["y"] = corrections_df["Q2"] / (
        2 * beam_energy * proton_mass * corrections_df["x"]
    )
    corrections_df["radiative_corrections"] = (
        corrections_df["Sig_Rad"] / corrections_df["Sig_Born"]
    )

    os.makedirs(flags.plots_directory, exist_ok=True)

    Q2_binning_for_plotting = np.linspace(1, 11, 11)

    fig = plt.figure(figsize=(12, 8))
    plt.hist(
        1 / corrections_df["radiative_corrections"],
        bins=np.linspace(0, 1.39, 51),
        density=True,
    )
    plt.xlabel("$\sigma_{Born}/\sigma_{rad}$")
    plt.ylabel("Normalized entries")
    plt.title(f"LD2 + {flags.target_name}")
    plt.savefig(flags.plots_directory + f"deltaRC_hist_y<1_{flags.plot_string}.png")

    make_2d_scatter_plot(
        xaxis_quantity=corrections_df["x"],
        yaxis_quantity=corrections_df["Q2"],
        zaxis_quantity=1 / corrections_df["radiative_corrections"],
        yaxis_quantity_name="$Q^2$",
        yaxis_binning=Q2_binning_for_plotting,
        xaxis_limits=(0, 1.05),
        yaxis_limits=(0, 1.39),
        xaxis_label="x",
        yaxis_label="$\sigma_{Born}/\sigma_{rad}$",
        title=f"LD2 + {flags.target_name}: y<1",
        save_path=flags.plots_directory
        + f"deltaRC_scatter_y<1_{flags.plot_string}.png",
    )

    make_2d_scatter_plot(
        xaxis_quantity=corrections_df["x"],
        yaxis_quantity=corrections_df["Q2"],
        zaxis_quantity=1 / corrections_df["radiative_corrections"],
        yaxis_quantity_name="$Q^2$",
        yaxis_binning=Q2_binning_for_plotting,
        cut=corrections_df["y"] < 0.85,
        draw_original_distribution=True,
        xaxis_limits=(0, 1.05),
        yaxis_limits=(0, 1.39),
        xaxis_label="x",
        yaxis_label="$\sigma_{Born}/\sigma_{rad}$",
        title=f"LD2 + {flags.target_name}: y<.85",
        save_path=flags.plots_directory
        + f"deltaRC_scatter_y<.85_{flags.plot_string}.png",
    )

    make_2d_scatter_plot(
        xaxis_quantity=corrections_df["x"],
        yaxis_quantity=corrections_df["Q2"],
        zaxis_quantity=1 / corrections_df["radiative_corrections"],
        yaxis_quantity_name="$Q^2$",
        yaxis_binning=Q2_binning_for_plotting,
        cut=corrections_df["y"] < 0.8,
        draw_original_distribution=True,
        xaxis_limits=(0, 1.05),
        yaxis_limits=(0, 1.39),
        xaxis_label="x",
        yaxis_label="$\sigma_{Born}/\sigma_{rad}$",
        title=f"LD2 + {flags.target_name}: y<.8",
        save_path=flags.plots_directory
        + f"deltaRC_scatter_y<.8_{flags.plot_string}.png",
    )

    make_2d_scatter_plot(
        xaxis_quantity=corrections_df["x"],
        yaxis_quantity=corrections_df["Q2"],
        zaxis_quantity=corrections_df["C_cor"],
        yaxis_quantity_name="$Q^2$",
        yaxis_binning=Q2_binning_for_plotting,
        xaxis_limits=(0, 1.05),
        yaxis_limits=(0.99, 1.15),
        xaxis_label="x",
        yaxis_label="$\delta_{CC}$",
        title=f"LD2 + {flags.target_name}",
        save_path=flags.plots_directory + f"deltaCC_scatter_y<1{flags.plot_string}.png",
        legend_location="upper right",
        legend_ncols=1,
        legend_fontsize=14,
    )

    make_2d_scatter_plot(
        xaxis_quantity=corrections_df["x"],
        yaxis_quantity=corrections_df["Q2"],
        zaxis_quantity=corrections_df["C_cor"],
        yaxis_quantity_name="$Q^2$",
        yaxis_binning=Q2_binning_for_plotting,
        cut=corrections_df["y"] < 0.85,
        draw_original_distribution=True,
        xaxis_limits=(0, 1.05),
        yaxis_limits=(0.99, 1.15),
        xaxis_label="x",
        yaxis_label="$\delta_{CC}$",
        title=f"LD2 + {flags.target_name}",
        save_path=flags.plots_directory
        + f"deltaCC_scatter_y<.85{flags.plot_string}.png",
        legend_location="upper right",
        legend_ncols=1,
        legend_fontsize=14,
    )

    make_2d_scatter_plot(
        xaxis_quantity=corrections_df["x"],
        yaxis_quantity=corrections_df["Q2"],
        zaxis_quantity=corrections_df["C_cor"],
        yaxis_quantity_name="$Q^2$",
        yaxis_binning=Q2_binning_for_plotting,
        cut=corrections_df["y"] < 0.8,
        draw_original_distribution=True,
        xaxis_limits=(0, 1.05),
        yaxis_limits=(0.99, 1.15),
        xaxis_label="x",
        yaxis_label="$\delta_{CC}$",
        title=f"LD2 + {flags.target_name}",
        save_path=flags.plots_directory
        + f"deltaCC_scatter_y<.8{flags.plot_string}.png",
        legend_location="upper right",
        legend_ncols=1,
        legend_fontsize=14,
    )
