import matplotlib.pyplot as plt
import matplotlib.colors as colors
from mpl_toolkits.axes_grid1 import make_axes_locatable
import mplhep as hep

hep.style.use(hep.style.CMS)
import numpy as np
import awkward as ak
import pandas as pd
from scipy.optimize import curve_fit
import os
import json

# Converting operator strings to operations on Awkward arrays
array_operator_dict = {
    ">": (lambda array, value: array > value),
    "<": (lambda array, value: array < value),
    ">=": (lambda array, value: array >= value),
    "<=": (lambda array, value: array <= value),
    "==": (lambda array, value: array == value),
    "!=": (lambda array, value: array != value),
}
num_sectors = 6


# Applying kinematic cuts to each electron based on ELECTRON_KINEMATIC_CUTS in the config file
def apply_kinematic_cuts(
    events,
    kinematic_cuts,
    save_plots=True,
    plots_directory=None,
    plot_title=None,
    log_file=None,
    number_of_initial_electrons=None,
):
    mask = np.ones(len(events), dtype=bool)
    y_cut_value, W_cut_value = None, None
    for cut in kinematic_cuts:
        variable_name, operation, cut_value = cut.split()
        if operation not in array_operator_dict:
            raise ValueError(f"Unsupported operation: {operation}")

        mask = (mask) & (
            array_operator_dict[operation](
                events["reconstructed"][variable_name], float(cut_value)
            )
        )
        if variable_name == "y":
            y_cut_value = float(cut_value)
        elif variable_name == "W":
            W_cut_value = float(cut_value)
    events["pass_reco"] = mask
    events["pass_kinematic"] = mask
    print(f"Have {ak.sum(events['pass_reco'])} events after kinematic cuts")
    if save_plots:
        # DIS kinematic boundaries in x-Q2 space, drawn from this call's own
        # y/W cut values so the plotted lines always match what was applied.
        M_p = 0.93827  # GeV
        beam_energy = 10.547  # GeV -- RG-E beam energy
        s = M_p**2 + 2 * M_p * beam_energy  # GeV^2, massless-electron approx

        x_line = np.linspace(1e-3, 1, 500)

        # y = Q^2 / (x*(s - M_p^2))  ->  Q^2(x) = y * (s - M_p^2) * x
        Q2_y_cut = y_cut_value * (s - M_p**2) * x_line

        # W^2 = M_p^2 + Q^2*(1-x)/x  ->  Q^2(x) = (W^2 - M_p^2) * x/(1-x)
        Q2_W_cut = (W_cut_value**2 - M_p**2) * x_line / (1 - x_line)

        x_bins = np.linspace(0, 1, num=50 + 1)
        Q2_bins = np.logspace(np.log10(1), np.log10(11), num=45 + 1, base=10.0)
        x_bins_fine = np.linspace(0, 1, num=200 + 1)
        Q2_bins_fine = np.logspace(np.log10(1), np.log10(11), num=200 + 1, base=10.0)

        def plot_x_Q2(x, Q2, bins, file_name, draw_cut_lines):
            fig = plt.figure(figsize=(12, 8))
            plt.hist2d(np.array(x), np.array(Q2), bins=bins, norm=colors.LogNorm())
            if draw_cut_lines:
                plt.plot(
                    x_line,
                    Q2_y_cut,
                    color="red",
                    linestyle="--",
                    linewidth=1.5,
                    label=f"$y={y_cut_value}$",
                )
                plt.plot(
                    x_line,
                    Q2_W_cut,
                    color="cyan",
                    linestyle="--",
                    linewidth=1.5,
                    label=f"$W={W_cut_value}$ GeV",
                )
                plt.legend(loc="upper left")
            plt.xlabel("x")
            plt.ylabel("$Q^2~(GeV^2)$")
            plt.colorbar()
            if plot_title is not None:
                plt.title(plot_title)
            plt.xlim(0, 1)
            plt.ylim(1, 11)
            if plots_directory is not None:
                plt.savefig(plots_directory + file_name)
            plt.close()

        plot_x_Q2(
            events["reconstructed"]["x"],
            events["reconstructed"]["Q2"],
            (x_bins, Q2_bins),
            "x_Q2_before_cuts.png",
            draw_cut_lines=True,
        )
        plot_x_Q2(
            events["reconstructed"]["x"],
            events["reconstructed"]["Q2"],
            (x_bins, Q2_bins),
            "x_Q2_before_cuts_nocutlines.png",
            draw_cut_lines=False,
        )
        plot_x_Q2(
            events["reconstructed"]["x"][mask],
            events["reconstructed"]["Q2"][mask],
            (x_bins_fine, Q2_bins_fine),
            "x_Q2_after_cuts_finebinning.png",
            draw_cut_lines=True,
        )
        plot_x_Q2(
            events["reconstructed"]["x"][mask],
            events["reconstructed"]["Q2"][mask],
            (x_bins, Q2_bins),
            "x_Q2_after_cuts.png",
            draw_cut_lines=True,
        )
    if log_file is not None:
        with open(log_file, "a") as f:
            f.write(f"Have {ak.sum(events['pass_reco'])} events after kinematic cuts\n")
    if log_file is not None:
        with open(log_file, "a") as f:
            f.write(
                f"Have {ak.sum(events['pass_reco'])/number_of_initial_electrons} fraction of events passing kinematic cuts\n"
            )
    return events


# Applying fiducial cuts to each electron based on ELECTRON_FIDUCIAL_CUTS in the config file
def apply_fiducial_cuts(
    events,
    fiducial_cuts,
    save_plots=True,
    plots_directory=None,
    plot_title=None,
    log_file=None,
    number_of_initial_electrons=None,
):

    PCAL_V_cut, PCAL_W_cut = None, None
    PCAL_fiducial_mask = np.ones(len(events), dtype=bool)
    DC_region1_cut, DC_region2_cut, DC_region3_cut = None, None, None
    DC_fiducial_mask = np.ones(len(events), dtype=bool)
    DC_region1_mask = np.ones(len(events), dtype=bool)
    DC_region2_mask = np.ones(len(events), dtype=bool)
    DC_region3_mask = np.ones(len(events), dtype=bool)

    for cut in fiducial_cuts:
        variable_name, operation, cut_value = cut.split()
        if variable_name == "PCAL_V":
            PCAL_fiducial_mask = (PCAL_fiducial_mask) & (
                array_operator_dict[operation](
                    events["reconstructed"][variable_name], float(cut_value)
                )
            )
            PCAL_V_cut = float(cut_value)
        elif variable_name == "PCAL_W":
            PCAL_fiducial_mask = (PCAL_fiducial_mask) & (
                array_operator_dict[operation](
                    events["reconstructed"][variable_name], float(cut_value)
                )
            )
            PCAL_W_cut = float(cut_value)
        elif variable_name == "DC_region1_edge":
            DC_region1_mask = array_operator_dict[operation](
                events["reconstructed"][variable_name], float(cut_value)
            )
            DC_region1_cut = float(cut_value)
        elif variable_name == "DC_region2_edge":
            DC_region2_mask = array_operator_dict[operation](
                events["reconstructed"][variable_name], float(cut_value)
            )
            DC_region2_cut = float(cut_value)
        elif variable_name == "DC_region3_edge":
            DC_region3_mask = array_operator_dict[operation](
                events["reconstructed"][variable_name], float(cut_value)
            )
            DC_region3_cut = float(cut_value)

    DC_fiducial_mask = (DC_region1_mask) & (DC_region2_mask) & (DC_region3_mask)

    if save_plots:

        ########################################################################
        ######################## Plotting the PCAL cuts ########################
        ########################################################################
        low_bin, high_bin, num_bins = (0, 30), (0, 0.35), (100, 100)
        fig, axs = plt.subplots(1, 2, figsize=(12, 6))
        _, _, _, mesh = axs[0].hist2d(
            np.array(events["reconstructed"]["PCAL_V"]),
            np.array(events["reconstructed"]["SF"]),
            bins=num_bins,
            range=(low_bin, high_bin),
            norm=colors.LogNorm(),
        )
        axs[0].set_ylabel("SF")
        axs[0].set_xlabel("PCAL V (cm)")
        if PCAL_V_cut is not None:
            axs[0].vlines(PCAL_V_cut, low_bin[0], low_bin[1], color="red")

        divider = make_axes_locatable(axs[0])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = fig.colorbar(mesh, cax=cax)

        _, _, _, mesh = axs[1].hist2d(
            np.array(events["reconstructed"]["PCAL_W"]),
            np.array(events["reconstructed"]["SF"]),
            bins=num_bins,
            range=(low_bin, high_bin),
            norm=colors.LogNorm(),
        )
        axs[1].set_ylabel("SF")
        axs[1].set_xlabel("PCAL W (cm)")
        if PCAL_W_cut is not None:
            axs[1].vlines(PCAL_W_cut, low_bin[0], low_bin[1], color="red")
        divider = make_axes_locatable(axs[1])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = fig.colorbar(mesh, cax=cax)
        plt.tight_layout()
        if plot_title is not None:
            plt.suptitle(plot_title, y=1.0)
        if plots_directory is not None:
            plt.savefig(plots_directory + "PCAL_W_V.png")

        low_bin, high_bin, num_bins = (0, 30), (0, 0.35), (100, 100)
        fig, axs = plt.subplots(1, 2, figsize=(12, 6))
        _, _, _, mesh = axs[0].hist2d(
            np.array(events["reconstructed"]["PCAL_U"]),
            np.array(events["reconstructed"]["SF"]),
            bins=num_bins,
            range=(low_bin, high_bin),
            norm=colors.LogNorm(),
        )
        axs[0].set_ylabel("SF")
        axs[0].set_xlabel("PCAL U (cm)")
        divider = make_axes_locatable(axs[0])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = fig.colorbar(mesh, cax=cax)

        _, _, _, mesh = axs[1].hist2d(
            np.array(events["reconstructed"]["PCAL_U"][PCAL_fiducial_mask]),
            np.array(events["reconstructed"]["SF"][PCAL_fiducial_mask]),
            bins=num_bins,
            range=(low_bin, high_bin),
            norm=colors.LogNorm(),
        )
        axs[1].set_ylabel("SF")
        axs[1].set_xlabel("PCAL U (cm)")
        divider = make_axes_locatable(axs[1])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = fig.colorbar(mesh, cax=cax)
        plt.tight_layout()
        if plot_title is not None:
            plt.suptitle(plot_title, y=1.0)
        if plots_directory is not None:
            plt.savefig(plots_directory + "PCAL_U.png")
        plt.close()

        ########################################################################
        ############## Plotting the DC cuts before fiducial cuts ##############
        ########################################################################
        region1_low_bin, region1_high_bin, region1_num_bins = (
            (-150, 150),
            (-150, 150),
            (250, 250),
        )
        region2_low_bin, region2_high_bin, region2_num_bins = (
            (-200, 200),
            (-200, 200),
            (250, 250),
        )
        region3_low_bin, region3_high_bin, region3_num_bins = (
            (-250, 250),
            (-250, 250),
            (250, 250),
        )

        fig, axs = plt.subplots(1, 3, figsize=(18, 6))
        _, _, _, mesh = axs[0].hist2d(
            np.array(events["reconstructed"]["DC_region1_x"]),
            np.array(events["reconstructed"]["DC_region1_y"]),
            bins=region1_num_bins,
            range=(region1_low_bin, region1_high_bin),
            norm=colors.LogNorm(),
        )
        axs[0].set_ylabel("y (cm)")
        axs[0].set_xlabel("x (cm)")
        axs[0].set_title("DC Region 1")

        divider = make_axes_locatable(axs[0])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = fig.colorbar(mesh, cax=cax)

        _, _, _, mesh = axs[1].hist2d(
            np.array(events["reconstructed"]["DC_region2_x"]),
            np.array(events["reconstructed"]["DC_region2_y"]),
            bins=region2_num_bins,
            range=(region2_low_bin, region2_high_bin),
            norm=colors.LogNorm(),
        )
        axs[1].set_ylabel("y (cm)")
        axs[1].set_xlabel("x (cm)")
        axs[1].set_title("DC Region 2")

        divider = make_axes_locatable(axs[1])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = fig.colorbar(mesh, cax=cax)

        _, _, _, mesh = axs[2].hist2d(
            np.array(events["reconstructed"]["DC_region3_x"]),
            np.array(events["reconstructed"]["DC_region3_y"]),
            bins=region3_num_bins,
            range=(region3_low_bin, region3_high_bin),
            norm=colors.LogNorm(),
        )
        axs[2].set_ylabel("y (cm)")
        axs[2].set_xlabel("x (cm)")
        axs[2].set_title("DC Region 3")

        divider = make_axes_locatable(axs[2])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = fig.colorbar(mesh, cax=cax)

        plt.tight_layout()
        if plot_title is not None:
            plt.suptitle(plot_title, y=1.0)
        if plots_directory is not None:
            plt.savefig(plots_directory + "DC_before_fiducialcuts.png")
        plt.close()

        ########################################################################
        ################ Plotting chi2/NDF vs distance to edge ################
        ########################################################################
        distance_to_edge_low_bin = 0
        distance_to_edge_high_bin = 20
        distance_to_edge_num_bins = 25
        bins = np.linspace(
            distance_to_edge_low_bin,
            distance_to_edge_high_bin,
            distance_to_edge_num_bins + 1,
        )
        bin_centers = 0.5 * (bins[:-1] + bins[1:])

        edge_cut_values = {
            "region1": DC_region1_cut,
            "region2": DC_region2_cut,
            "region3": DC_region3_cut,
        }
        fig, axs = plt.subplots(1, 3, figsize=(24, 7))

        for region_i in range(3):
            max_value = 0
            for sector in range(num_sectors):
                sector_mask = np.array(events["reconstructed"]["sector"] == sector + 1)

                distance_to_edge = np.array(
                    events["reconstructed"][f"DC_region{region_i+1}_edge"][sector_mask]
                )
                chi2 = np.array(events["reconstructed"]["chi2"][sector_mask])
                ndf = np.array(events["reconstructed"]["NDF"][sector_mask])
                chi2_per_ndf = chi2 / ndf
                bin_indices = np.digitize(distance_to_edge, bins) - 1

                bin_means = []
                for i in range(distance_to_edge_num_bins):
                    values_in_bin = chi2_per_ndf[bin_indices == i]
                    if len(values_in_bin) > 0:
                        bin_means.append(np.mean(values_in_bin))
                    else:
                        bin_means.append(np.nan)

                axs[region_i].scatter(
                    bin_centers, bin_means, label=f"Sector {sector + 1}"
                )
                bin_means = np.array(bin_means)[~np.isnan(bin_means)]
                max_bin_means = max(bin_means)
                if max_bin_means > max_value:
                    max_value = max_bin_means
            if edge_cut_values[f"region{region_i+1}"] is not None:
                axs[region_i].vlines(
                    edge_cut_values[f"region{region_i+1}"], 0, max_value, color="red"
                )
            axs[region_i].set_title(f"DC Region {region_i+1}")
            axs[region_i].set_xlabel("Distance to Edge (cm)")
            axs[region_i].set_ylabel("Average χ²/NDF")
            axs[region_i].legend(ncols=2, loc="upper right", columnspacing=0.8)
            axs[region_i].grid(True)
        plt.tight_layout()
        if plot_title is not None:
            plt.suptitle(plot_title, y=1.0)
        if plots_directory is not None:
            plt.savefig(plots_directory + "DC_chi2NDF.png")
        plt.close()

        ########################################################################
        ############### Plotting the DC cuts after fiducial cuts ##############
        ########################################################################
        region1_low_bin, region1_high_bin, region1_num_bins = (
            (-150, 150),
            (-150, 150),
            (250, 250),
        )
        region2_low_bin, region2_high_bin, region2_num_bins = (
            (-200, 200),
            (-200, 200),
            (250, 250),
        )
        region3_low_bin, region3_high_bin, region3_num_bins = (
            (-250, 250),
            (-250, 250),
            (250, 250),
        )

        fig, axs = plt.subplots(1, 3, figsize=(18, 6))
        _, _, _, mesh = axs[0].hist2d(
            np.array(events["reconstructed"]["DC_region1_x"][DC_fiducial_mask]),
            np.array(events["reconstructed"]["DC_region1_y"][DC_fiducial_mask]),
            bins=region1_num_bins,
            range=(region1_low_bin, region1_high_bin),
            norm=colors.LogNorm(),
        )
        axs[0].set_ylabel("y (cm)")
        axs[0].set_xlabel("x (cm)")
        axs[0].set_title("DC Region 1")

        divider = make_axes_locatable(axs[0])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = fig.colorbar(mesh, cax=cax)

        _, _, _, mesh = axs[1].hist2d(
            np.array(events["reconstructed"]["DC_region2_x"][DC_fiducial_mask]),
            np.array(events["reconstructed"]["DC_region2_y"][DC_fiducial_mask]),
            bins=region2_num_bins,
            range=(region2_low_bin, region2_high_bin),
            norm=colors.LogNorm(),
        )
        axs[1].set_ylabel("y (cm)")
        axs[1].set_xlabel("x (cm)")
        axs[1].set_title("DC Region 2")

        divider = make_axes_locatable(axs[1])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = fig.colorbar(mesh, cax=cax)

        _, _, _, mesh = axs[2].hist2d(
            np.array(events["reconstructed"]["DC_region3_x"][DC_fiducial_mask]),
            np.array(events["reconstructed"]["DC_region3_y"][DC_fiducial_mask]),
            bins=region3_num_bins,
            range=(region3_low_bin, region3_high_bin),
            norm=colors.LogNorm(),
        )
        axs[2].set_ylabel("y (cm)")
        axs[2].set_xlabel("x (cm)")
        axs[2].set_title("DC Region 3")

        divider = make_axes_locatable(axs[2])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = fig.colorbar(mesh, cax=cax)

        plt.tight_layout()
        if plot_title is not None:
            plt.suptitle(plot_title, y=1.0)
        if plots_directory is not None:
            plt.savefig(plots_directory + "DC_after_fiducialcuts.png")
        plt.close()

    fiducial_cuts = (PCAL_fiducial_mask) & (DC_fiducial_mask)

    if log_file is not None:
        with open(log_file, "a") as f:
            f.write(
                f"Have {ak.sum(PCAL_fiducial_mask)/number_of_initial_electrons} fraction of events after PCAL fiducial cuts\n"
            )
            f.write(
                f"Have {ak.sum(DC_fiducial_mask)/number_of_initial_electrons} fraction of events after DC fiducial cuts\n"
            )
            f.write(
                f"Have {ak.sum(fiducial_cuts)/number_of_initial_electrons} fraction of events after PCAL and DC fiducial cuts\n"
            )
            f.write(
                f"Have {ak.sum(DC_region1_mask)/number_of_initial_electrons} fraction of events after DC region 1 fiducial cut\n"
            )
            f.write(
                f"Have {ak.sum(DC_region2_mask)/number_of_initial_electrons} fraction of events after DC region 2 fiducial cut\n"
            )
            f.write(
                f"Have {ak.sum(DC_region3_mask)/number_of_initial_electrons} fraction of events after DC region 3 fiducial cut\n"
            )

    events["pass_reco"] = (fiducial_cuts) & (events["pass_reco"])

    print(f"Have {ak.sum(events['pass_reco'])} events after fiducial cuts")
    if log_file is not None:
        with open(log_file, "a") as f:
            f.write(
                f"Have {ak.sum(events['pass_reco'])} pass reco events after fiducial cuts\n"
            )
        with open(log_file, "a") as f:
            f.write(
                f"Have {ak.sum(events['pass_reco'])} fraction of events after PCAL and DC fiducial cuts and kinematic cuts\n"
            )

    events["pass_fiducial"] = fiducial_cuts
    return events


"""
Applying a cut on ECIN vs EPCAL.
Procedure:
1. In each sector, bin in momentum
2. In each momentum bin, make slices in SF(ECin) and make a histogram of SF(PCAL). Then fit each with a Gaussian
3. Take the mean and sigma from each Gaussian fit, and fit mu(SF PCAL)-2.5sigma(SF PCAL) in each SF(ECin) bin with a line
4. Only keep electrons that are above that line.

This function has a develop and apply mode. In the develop mode, the fits are made and the results are saved to a json.
In apply mode, the fit parameters are simply read from the json and applied.

"""


def apply_partial_sampling_fraction_cut(
    events,
    develop_cuts=False,
    cut_params_path=None,
    is_simulation=False,
    save_plots=True,
    plots_directory=None,
    plot_title=None,
    log_file=None,
    number_of_initial_electrons=None,
):
    momentum_bin_edges = [0, 2, 3, 4, 5, 6, 7, 8, 9, 12]

    partial_sampling_fraction_directory = None
    if plots_directory is not None:
        partial_sampling_fraction_directory = (
            plots_directory + "/partial_sampling_fraction/"
        )
        os.makedirs(partial_sampling_fraction_directory, exist_ok=True)

    if develop_cuts:
        events_for_fitting = events[events["pass_fiducial"]]
        all_sector_fits = {}

        for sector in range(num_sectors):
            sector_cut = events_for_fitting["reconstructed"]["sector"] == (sector + 1)
            data_in_sector = events_for_fitting["reconstructed"][sector_cut]

            sector_fits = _fit_sector_momentum_bins(
                data_in_sector,
                sector + 1,
                is_simulation,
                momentum_bin_edges,
                partial_sampling_fraction_directory,
                plot_title,
            )
            all_sector_fits[str(sector + 1)] = sector_fits

            if log_file is not None:
                with open(log_file, "a") as f:
                    f.write(f"\nSector {sector+1} partial SF linear fit parameters:\n")
                    a_row = f"$a_{{\\text{{sector{sector+1}}}}}$"
                    b_row = f"$b_{{\\text{{sector{sector+1}}}}}$"
                    for fit in sector_fits:
                        a_row += f" & {fit['slope']:.6f}"
                        b_row += f" & {fit['intercept']:.6f}"
                    f.write(a_row + " \\\\\n")
                    f.write(b_row + " \\\\\n")

        cut_params = {
            "momentum_bin_edges": momentum_bin_edges,
            "sectors": all_sector_fits,
        }

        if cut_params_path is not None:
            with open(cut_params_path, "w") as f:
                json.dump(cut_params, f, indent=2)

    else:
        if cut_params_path is None:
            raise ValueError("cut_params_path is required when develop_cuts=False")
        with open(cut_params_path, "r") as f:
            cut_params = json.load(f)

        if partial_sampling_fraction_directory is not None:
            events_for_plotting = events[events["pass_fiducial"]]
            for sector_str, sector_fits in cut_params["sectors"].items():
                sector = int(sector_str)
                sector_cut = events_for_plotting["reconstructed"]["sector"] == sector
                data_in_sector = events_for_plotting["reconstructed"][sector_cut]
                if save_plots:
                    _plot_sector_summary_from_params(
                        data_in_sector,
                        sector,
                        sector_fits,
                        partial_sampling_fraction_directory,
                        plot_title,
                    )

    return _apply_partial_sf_mask(
        events, cut_params, log_file, number_of_initial_electrons
    )


def _line_equation(x, m, b):
    return m * x + b


def _fit_sector_momentum_bins(
    data_in_sector,
    sector,
    is_simulation,
    momentum_bin_edges,
    plots_directory,
    plot_title,
):
    """Fit (mu - 2.5*sigma) vs SF(ECIN) per momentum bin for one sector.
    Returns list of dicts: bin_index, low/high_momentum_edge, slope, intercept."""
    ECIN_SF_in_sector = np.array(data_in_sector["E_ECIN"] / data_in_sector["p"])
    PCAL_SF_in_sector = np.array(data_in_sector["E_PCAL"] / data_in_sector["p"])

    save_plots = plots_directory is not None
    sector_fits = []

    if save_plots:
        fig_all_momenta, axs_all_momenta = plt.subplots(3, 3, figsize=(24, 24))
        axs_all_momenta = axs_all_momenta.flatten()

    for i, low_momentum_edge in enumerate(momentum_bin_edges[:-1]):
        high_momentum_edge = momentum_bin_edges[i + 1]
        momentum_cut = (data_in_sector["p"] > low_momentum_edge) & (
            data_in_sector["p"] <= high_momentum_edge
        )

        if save_plots:
            axs_all_momenta[i].set_xlabel("$E_{ECIN}$/p")
            axs_all_momenta[i].set_ylabel("$E_{PCAL}$/p")
            axs_all_momenta[i].set_title(
                f"{low_momentum_edge} GeV < p < {high_momentum_edge} GeV"
            )

        if np.sum(momentum_cut) == 0:
            continue

        ECIN_SF_slice = ECIN_SF_in_sector[momentum_cut]
        PCAL_SF_slice = PCAL_SF_in_sector[momentum_cut]

        ECIN_bins = (
            np.linspace(0.08, 0.15, 11)
            if is_simulation
            else np.linspace(0.05, 0.15, 11)
        )
        PCAL_bins = np.linspace(0.0, 0.25, 51)
        PCAL_bin_centers = (PCAL_bins[1:] + PCAL_bins[:-1]) / 2

        ECIN_bin_means, ECIN_bin_sigma, valid_ECIN_bin_centers = [], [], []

        if save_plots:
            fig_gaussians, axs_gaussians = plt.subplots(2, 5, figsize=(28, 18))
            axs_gaussians = axs_gaussians.flatten()

        for j, low_ECIN_edge in enumerate(ECIN_bins[:-1]):
            high_ECIN_edge = ECIN_bins[j + 1]
            ECIN_mask = (ECIN_SF_slice > low_ECIN_edge) & (
                ECIN_SF_slice <= high_ECIN_edge
            )
            PCAL_SF_in_ECIN_slice = PCAL_SF_slice[ECIN_mask]

            counts, _ = np.histogram(PCAL_SF_in_ECIN_slice, bins=PCAL_bins)
            if save_plots:
                axs_gaussians[j].hist(PCAL_SF_in_ECIN_slice, bins=PCAL_bins)

            if np.sum(counts) < 100:
                print(
                    f"Sector {sector}, {low_momentum_edge}-{high_momentum_edge} GeV: "
                    f"NO FIT for ECIN slice {low_ECIN_edge:.3f}-{high_ECIN_edge:.3f} "
                    f"(only {int(np.sum(counts))} events, need >= 100) — skipping this slice"
                )
                continue

            if high_ECIN_edge < 0.1:
                fit_mask = PCAL_bin_centers > 0.11
                p0 = (
                    len(PCAL_SF_in_ECIN_slice),
                    np.mean(PCAL_SF_in_ECIN_slice[PCAL_SF_in_ECIN_slice > 0.11]),
                    np.std(PCAL_SF_in_ECIN_slice[PCAL_SF_in_ECIN_slice > 0.11]) ** 2,
                )
                popt, _ = curve_fit(
                    gaus, PCAL_bin_centers[fit_mask], counts[fit_mask], p0=p0
                )
            else:
                p0 = (
                    len(PCAL_SF_in_ECIN_slice),
                    np.mean(PCAL_SF_in_ECIN_slice),
                    np.std(PCAL_SF_in_ECIN_slice) ** 2,
                )
                popt, _ = curve_fit(gaus, PCAL_bin_centers, counts, p0=p0)

            if save_plots:
                axs_gaussians[j].plot(PCAL_bin_centers, gaus(PCAL_bin_centers, *popt))
                axs_gaussians[j].set_title(
                    f"{round(low_ECIN_edge,3)} < SF(ECIN) < {round(high_ECIN_edge,3)}"
                )
                axs_gaussians[j].set_xlabel("SF(PCAL)")

            ECIN_bin_means.append(popt[1])
            ECIN_bin_sigma.append(np.sqrt(popt[2]))
            valid_ECIN_bin_centers.append((high_ECIN_edge + low_ECIN_edge) / 2)

        if save_plots:
            if plot_title is not None:
                fig_gaussians.suptitle(
                    plot_title
                    + f"\nSector {sector}, {low_momentum_edge} GeV<p<{high_momentum_edge} GeV",
                    y=1.0,
                )
            fig_gaussians.tight_layout()
            fig_gaussians.savefig(
                plots_directory
                + f"gaussianfits_sector{sector}_momentum{low_momentum_edge}_{high_momentum_edge}.png"
            )
            plt.close(fig_gaussians)

        valid_ECIN_bin_centers = np.array(valid_ECIN_bin_centers)
        if len(valid_ECIN_bin_centers) == 0:
            print(
                f"Sector {sector}, {low_momentum_edge}-{high_momentum_edge} GeV: "
                "NO FIT — no ECIN slices had enough statistics; every event in this "
                "sector/momentum bin will FAIL the partial SF cut"
            )
            continue

        target = np.asarray(ECIN_bin_means) - 2.5 * np.asarray(ECIN_bin_sigma)
        try:
            popt, _ = curve_fit(_line_equation, valid_ECIN_bin_centers, target)
        except (TypeError, RuntimeError, ValueError):
            print(
                f"Sector {sector}, {low_momentum_edge}-{high_momentum_edge} GeV: "
                "NO FIT — linear fit of SF(ECIN) vs SF(PCAL) failed; every event in "
                "this sector/momentum bin will FAIL the partial SF cut"
            )
            continue

        sector_fits.append(
            {
                "bin_index": i,
                "low_momentum_edge": float(low_momentum_edge),
                "high_momentum_edge": float(high_momentum_edge),
                "slope": float(popt[0]),
                "intercept": float(popt[1]),
            }
        )

        if save_plots:
            fig_scatter = plt.figure(figsize=(12, 8))
            plt.scatter(valid_ECIN_bin_centers, target)
            plt.plot(
                valid_ECIN_bin_centers,
                _line_equation(valid_ECIN_bin_centers, *popt),
                color="red",
                linestyle="dashed",
            )
            plt.xlabel("SF(ECIN)")
            plt.ylabel(r"$\mu$ SF(PCAL) - 2.5$\sigma$ SF(PCAL)")
            if plot_title is not None:
                plt.title(
                    plot_title
                    + f"\nSector {sector}, {low_momentum_edge} GeV<p<{high_momentum_edge} GeV",
                    y=1.0,
                )
            fig_scatter.savefig(
                plots_directory
                + f"sector{sector}_momentum{low_momentum_edge}_{high_momentum_edge}.png"
            )
            plt.close(fig_scatter)

            _, bins, _, mesh = axs_all_momenta[i].hist2d(
                ECIN_SF_slice,
                PCAL_SF_slice,
                bins=(100, 100),
                range=[(0, 0.2), (0, 0.25)],
                norm=colors.LogNorm(),
            )
            ECIN_bin_centers = (bins[1:] + bins[:-1]) / 2
            divider = make_axes_locatable(axs_all_momenta[i])
            cax = divider.append_axes("right", size="5%", pad=0.05)
            fig_all_momenta.colorbar(mesh, cax=cax)
            axs_all_momenta[i].plot(
                ECIN_bin_centers, _line_equation(ECIN_bin_centers, *popt), color="red"
            )

    if save_plots:
        if plot_title is not None:
            fig_all_momenta.suptitle(plot_title + f"\nSector {sector}", y=0.98)
        fig_all_momenta.tight_layout()
        fig_all_momenta.savefig(
            plots_directory + f"partial_sampling_sector{sector}.png"
        )
        plt.close(fig_all_momenta)

    return sector_fits


def _plot_sector_summary_from_params(
    data_in_sector, sector, sector_fits, plots_directory, plot_title
):
    """Apply-mode diagnostic: hist2d per momentum bin with the *loaded* fit line overlaid."""
    ECIN_SF_in_sector = np.array(data_in_sector["E_ECIN"] / data_in_sector["p"])
    PCAL_SF_in_sector = np.array(data_in_sector["E_PCAL"] / data_in_sector["p"])

    fig_all_momenta, axs_all_momenta = plt.subplots(3, 3, figsize=(24, 24))
    axs_all_momenta = axs_all_momenta.flatten()

    for fit in sector_fits:
        i = fit["bin_index"]
        low, high = fit["low_momentum_edge"], fit["high_momentum_edge"]
        momentum_cut = (data_in_sector["p"] > low) & (data_in_sector["p"] <= high)

        axs_all_momenta[i].set_xlabel("$E_{ECIN}$/p")
        axs_all_momenta[i].set_ylabel("$E_{PCAL}$/p")
        axs_all_momenta[i].set_title(f"{low} GeV < p < {high} GeV")

        if np.sum(momentum_cut) == 0:
            continue

        _, bins, _, mesh = axs_all_momenta[i].hist2d(
            ECIN_SF_in_sector[momentum_cut],
            PCAL_SF_in_sector[momentum_cut],
            bins=(100, 100),
            range=[(0, 0.2), (0, 0.25)],
            norm=colors.LogNorm(),
        )
        ECIN_bin_centers = (bins[1:] + bins[:-1]) / 2
        divider = make_axes_locatable(axs_all_momenta[i])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        fig_all_momenta.colorbar(mesh, cax=cax)
        axs_all_momenta[i].plot(
            ECIN_bin_centers,
            _line_equation(ECIN_bin_centers, fit["slope"], fit["intercept"]),
            color="red",
        )

    if plot_title is not None:
        fig_all_momenta.suptitle(plot_title + f"\nSector {sector}", y=0.98)
    fig_all_momenta.tight_layout()
    fig_all_momenta.savefig(plots_directory + f"partial_sampling_sector{sector}.png")
    plt.close(fig_all_momenta)


def _apply_partial_sf_mask(
    events, cut_params, log_file=None, number_of_initial_electrons=None
):
    sector_arr = np.array(events["reconstructed"]["sector"])
    p_arr = np.array(events["reconstructed"]["p"])
    ECIN_SF = np.array(events["reconstructed"]["E_ECIN"] / events["reconstructed"]["p"])
    PCAL_SF = np.array(events["reconstructed"]["E_PCAL"] / events["reconstructed"]["p"])

    partial_SF_mask = np.zeros(len(events["reconstructed"]), dtype=bool)

    for sector_str, sector_fits in cut_params["sectors"].items():
        sector = int(sector_str)
        for fit in sector_fits:
            current_mask = (
                (sector_arr == sector)
                & (p_arr > fit["low_momentum_edge"])
                & (p_arr <= fit["high_momentum_edge"])
                & (PCAL_SF > _line_equation(ECIN_SF, fit["slope"], fit["intercept"]))
            )
            partial_SF_mask |= current_mask

    events["pass_reco"] = events["pass_reco"] & partial_SF_mask
    events["pass_partial_SF"] = partial_SF_mask

    print(f"Have {ak.sum(events['pass_reco'])} events after partial SF cuts")
    if log_file is not None:
        with open(log_file, "a") as f:
            f.write(
                f"\nHave {ak.sum(events['pass_reco'])} pass reco events after partial SF cuts\n"
            )
            if number_of_initial_electrons is not None:
                f.write(
                    f"Have {ak.sum(partial_SF_mask)/number_of_initial_electrons} fraction of events after partial SF cuts\n"
                )
                f.write(
                    f"Have {ak.sum((events['pass_fiducial']) & (partial_SF_mask))/number_of_initial_electrons} "
                    "fraction of events after fiducial, and partial SF cuts\n"
                )

    return events


def gaus(x, a, mu, sigma_squared):
    return a * np.exp(-((x - mu) ** 2) / (2 * sigma_squared))


def sf_fit_function(x, a, b, c):
    return a + b / x + c / (x * x)


def sf_gaussians_by_sector(
    sampling_fractions_in_sector,
    xaxis_in_sector,
    xaxis_bins_in_sector,
    sector_number,
    SF_bins,
    xaxis_name,
    save_plots=True,
    plots_directory=None,
    plot_title=None,
):
    sf_fit_data = {
        "bin_low": [],
        "bin_high": [],
        "bin_center": [],
        "mu": [],
        "sigma": [],
    }
    low_sf_bin, high_sf_bin = SF_bins[0], SF_bins[1]
    fig, axs = plt.subplots(10, 10, figsize=(45, 55))
    fig.subplots_adjust(hspace=0.1, wspace=0.1)
    axs = axs.flatten()
    for i, lower_bin_edge in enumerate(xaxis_bins_in_sector):
        if i == len(xaxis_bins_in_sector) - 1:
            break
        upper_bin_edge = xaxis_bins_in_sector[i + 1]
        xaxis_bin_mask = (xaxis_in_sector > lower_bin_edge) & (
            xaxis_in_sector < upper_bin_edge
        )

        counts, bins, _ = axs[i].hist(
            sampling_fractions_in_sector[xaxis_bin_mask],
            bins=100,
            range=(low_sf_bin, high_sf_bin),
        )

        slice_center = (lower_bin_edge + upper_bin_edge) / 2
        bin_centers = (bins[:-1] + bins[1:]) / 2

        mean_in_bin = np.mean(sampling_fractions_in_sector[xaxis_bin_mask])
        std_in_bin = np.std(sampling_fractions_in_sector[xaxis_bin_mask])
        if slice_center < 0.5:
            sf_mask = (bin_centers > 0.18) & (bin_centers < 0.3)
        else:
            sf_mask = (bin_centers > 0.22) & (bin_centers < 0.3)
        try:
            popt, pcov = curve_fit(
                gaus,
                bin_centers[sf_mask],
                counts[sf_mask],
                p0=(
                    len(sampling_fractions_in_sector[xaxis_bin_mask]),
                    mean_in_bin,
                    std_in_bin * std_in_bin,
                ),
            )
            sf_fit_data["mu"].append(popt[1])
            sf_fit_data["sigma"].append(np.sqrt(popt[2]))
            axs[i].plot(bin_centers[sf_mask], gaus(bin_centers[sf_mask], *popt))
        except Exception:
            print(
                f"Sector {sector_number}: NO FIT for {xaxis_name} slice "
                f"{round(lower_bin_edge,3)}-{round(upper_bin_edge,3)} GeV — "
                "Gaussian fit of SF failed; this bin will be dropped from the "
                "mu(Edep)/sigma(Edep) fit"
            )
            sf_fit_data["mu"].append(None)
            sf_fit_data["sigma"].append(None)
        sf_fit_data["bin_low"].append(lower_bin_edge)
        sf_fit_data["bin_high"].append(upper_bin_edge)
        sf_fit_data["bin_center"].append((upper_bin_edge + lower_bin_edge) / 2)

        axs[i].set_xlabel("SF", fontsize=10)
        axs[i].set_title(
            f"{round(lower_bin_edge,3)} GeV < {xaxis_name} < {round(upper_bin_edge,3)} GeV",
            fontsize=10,
        )
        axs[i].tick_params(axis="both", which="major", labelsize=10)
    fig.tight_layout()

    if save_plots:
        if plot_title is not None:
            fig.suptitle(f"Sector {sector_number}", y=1.01)
        if plots_directory is not None:
            plt.savefig(plots_directory + f"sector{sector_number}_gaussian_fit.png")
    plt.close()
    return pd.DataFrame(sf_fit_data)


def _build_sector_arrays(
    electrons, pass_fiducial, save_plots, plots_directory, plot_title
):
    """Builds per-sector edep/SF arrays (fiducial-filtered) and optionally plots the raw 2D hist."""
    low_edep_bin, high_edep_bin = 0.12, 2.0
    low_sf_bin, high_sf_bin = 0.1, 0.35

    sampling_fraction_by_sector, edep_by_sector, edep_bins_by_sector = [], [], []

    if save_plots:
        fig, axs = plt.subplots(3, 2, figsize=(18, 18))
        axs = axs.flatten()

    for sector in range(num_sectors):
        sector_cut = (electrons["sector"] == (sector + 1)) & (pass_fiducial)
        total_ecal_energy = np.array(electrons["total_ecal_energy"][sector_cut])
        sampling_fraction = np.array(electrons["SF"][sector_cut])
        sampling_fraction_by_sector.append(sampling_fraction)
        edep_by_sector.append(total_ecal_energy)

        if save_plots:
            _, edep_bins, _, mesh = axs[sector].hist2d(
                total_ecal_energy,
                sampling_fraction,
                bins=(100, 100),
                range=[(low_edep_bin, high_edep_bin), (low_sf_bin, high_sf_bin)],
                norm=colors.LogNorm(),
            )
            edep_bins_by_sector.append(edep_bins)
            axs[sector].set_xlabel("$E_{dep}$ (GeV)")
            axs[sector].set_ylabel("$(E_{PCAL}+E_{ECIN}+E_{ECOUT})/P$")
            axs[sector].set_title(f"Sector {sector+1}")
            divider = make_axes_locatable(axs[sector])
            cax = divider.append_axes("right", size="5%", pad=0.05)
            fig.colorbar(mesh, cax=cax)
        else:
            # still need bin edges for fitting even without plotting
            edep_bins_by_sector.append(
                np.histogram_bin_edges(
                    total_ecal_energy, bins=100, range=(low_edep_bin, high_edep_bin)
                )
            )

    if save_plots:
        fig.tight_layout()
        if plot_title is not None:
            plt.suptitle(plot_title, y=1.0)
        if plots_directory is not None:
            plt.savefig(plots_directory + "sector_SF_without_fit.png")
        plt.close()

    return sampling_fraction_by_sector, edep_by_sector, edep_bins_by_sector


def _fit_mu_sigma_vs_edep(
    sf_fit_data_df, sector_number, save_plots, plots_directory, plot_title
):
    none_mask = sf_fit_data_df["mu"].notna() & sf_fit_data_df["sigma"].notna()
    bin_centers = sf_fit_data_df["bin_center"][none_mask].tolist()

    popt_mu, _ = curve_fit(
        sf_fit_function,
        bin_centers,
        sf_fit_data_df["mu"][none_mask].tolist(),
        p0=(0.2, -0.03, -0.001),
    )
    popt_sigma, _ = curve_fit(
        sf_fit_function,
        bin_centers,
        sf_fit_data_df["sigma"][none_mask].tolist(),
        p0=(0.2, -0.03, -0.001),
    )

    if save_plots:
        fig, axs = plt.subplots(1, 2, figsize=(15, 6))
        fig.subplots_adjust(hspace=0.1, wspace=0.3)
        axs[0].scatter(bin_centers, sf_fit_data_df["mu"][none_mask].tolist())
        axs[0].set_xlabel("bin center (GeV)")
        axs[0].set_ylabel("SF $\\mu$")
        axs[0].plot(
            bin_centers, sf_fit_function(np.array(bin_centers), *popt_mu), color="red"
        )

        axs[1].scatter(bin_centers, sf_fit_data_df["sigma"][none_mask].tolist())
        axs[1].set_xlabel("bin center (GeV)")
        axs[1].set_ylabel("SF $\\sigma$")
        axs[1].plot(
            bin_centers,
            sf_fit_function(np.array(bin_centers), *popt_sigma),
            color="red",
        )

        if plot_title is not None:
            plt.suptitle(plot_title + f",Sector {sector_number}", y=1.0)
        if plots_directory is not None:
            plt.savefig(plots_directory + f"SF_mu_fits_sector{sector_number}.png")
        plt.close()

    return popt_mu, popt_sigma


def _plot_sf_with_fit(
    edep_by_sector,
    sampling_fraction_by_sector,
    popt_mu_by_sector,
    popt_sigma_by_sector,
    plots_directory,
    plot_title,
):
    fig, axs = plt.subplots(3, 2, figsize=(18, 18))
    axs = axs.flatten()

    for sector in range(num_sectors):
        popt_mu = popt_mu_by_sector[sector]
        popt_sigma = popt_sigma_by_sector[sector]

        hist, edep_bins, sf_bins, mesh = axs[sector].hist2d(
            edep_by_sector[sector],
            sampling_fraction_by_sector[sector],
            bins=(100, 100),
            range=[(0, 2.5), (0.05, 0.35)],
            norm=colors.LogNorm(),
        )
        edep_bin_centers = (edep_bins[:-1] + edep_bins[1:]) / 2

        axs[sector].set_xlabel("$E_{dep}$ (GeV)")
        axs[sector].set_ylabel("$(E_{PCAL}+E_{ECIN}+E_{ECOUT})/P$")
        axs[sector].set_title(f"Sector {sector+1}")
        divider = make_axes_locatable(axs[sector])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(mesh, cax=cax)

        mu_curve = sf_fit_function(np.array(edep_bin_centers.tolist()), *popt_mu)
        sigma_curve = sf_fit_function(np.array(edep_bin_centers.tolist()), *popt_sigma)
        axs[sector].plot(edep_bin_centers.tolist(), mu_curve, color="black")
        axs[sector].plot(
            edep_bin_centers.tolist(), mu_curve + 3.5 * sigma_curve, color="red"
        )
        axs[sector].plot(
            edep_bin_centers.tolist(), mu_curve - 3.5 * sigma_curve, color="red"
        )
        axs[sector].set_xlim(0, 2.5)
        axs[sector].set_ylim(0.05, 0.35)

    fig.tight_layout()
    if plot_title is not None:
        plt.suptitle(plot_title, y=1.0)
    if plots_directory is not None:
        plt.savefig(plots_directory + "sector_SF_with_fit.png")
    plt.close()


"""
Tightening the SF vs ECAL Edep fit to mu +- 3.5 sigma from 5 sigma
To do this, we bin in SF vs Edep and in each Edep bin, we fit the SF with a Gaussian 
We then fit the Gaussian mean and sigma with a + b / x + c / (x * x) and remove events falling ouside +- 3.5 sigma
This is done separately for each sector
There is a develop_cuts mode where these fits are done and saved to a json file,
by default the cuts are read from the json file and applied without redoing the fits.
"""


def apply_sampling_fraction_cut(
    events,
    develop_cuts=False,
    cut_params_path=None,
    save_plots=True,
    plots_directory=None,
    plot_title=None,
    log_file=None,
    number_of_initial_electrons=None,
):
    events["reconstructed"] = ak.with_field(
        events["reconstructed"],
        events["reconstructed"]["E_PCAL"]
        + events["reconstructed"]["E_ECIN"]
        + events["reconstructed"]["E_ECOUT"],
        "total_ecal_energy",
    )
    electrons = events["reconstructed"]

    if develop_cuts:
        sampling_fraction_by_sector, edep_by_sector, edep_bins_by_sector = (
            _build_sector_arrays(
                electrons,
                events["pass_fiducial"],
                save_plots,
                plots_directory,
                plot_title,
            )
        )

        popt_mu_by_sector, popt_sigma_by_sector = [], []
        for sector in range(num_sectors):
            sector_number = sector + 1
            print(f"Fitting SF vs. Edep for sector {sector_number}")
            sf_df = sf_gaussians_by_sector(
                sampling_fraction_by_sector[sector],
                edep_by_sector[sector],
                edep_bins_by_sector[sector],
                sector_number,
                SF_bins=(0.1, 0.35),
                xaxis_name="$E_{dep}$",
                save_plots=save_plots,
                plots_directory=plots_directory,
                plot_title=plot_title,
            )
            popt_mu, popt_sigma = _fit_mu_sigma_vs_edep(
                sf_df, sector_number, save_plots, plots_directory, plot_title
            )
            popt_mu_by_sector.append(popt_mu)
            popt_sigma_by_sector.append(popt_sigma)

        if save_plots:
            _plot_sf_with_fit(
                edep_by_sector,
                sampling_fraction_by_sector,
                popt_mu_by_sector,
                popt_sigma_by_sector,
                plots_directory,
                plot_title,
            )

        cut_params = {
            "sectors": {
                str(sector + 1): {
                    "mu": popt_mu_by_sector[sector].tolist(),
                    "sigma": popt_sigma_by_sector[sector].tolist(),
                }
                for sector in range(num_sectors)
            }
        }
        if cut_params_path is not None:
            with open(cut_params_path, "w") as f:
                json.dump(cut_params, f, indent=2)

    else:
        if cut_params_path is None:
            raise ValueError("cut_params_path is required when develop_cuts=False")
        with open(cut_params_path, "r") as f:
            cut_params = json.load(f)

        popt_mu_by_sector = [
            np.array(cut_params["sectors"][str(s + 1)]["mu"])
            for s in range(num_sectors)
        ]
        popt_sigma_by_sector = [
            np.array(cut_params["sectors"][str(s + 1)]["sigma"])
            for s in range(num_sectors)
        ]

        if save_plots and plots_directory is not None:
            sampling_fraction_by_sector, edep_by_sector, _ = _build_sector_arrays(
                electrons,
                events["pass_fiducial"],
                save_plots,
                plots_directory,
                plot_title,
            )
            _plot_sf_with_fit(
                edep_by_sector,
                sampling_fraction_by_sector,
                popt_mu_by_sector,
                popt_sigma_by_sector,
                plots_directory,
                plot_title,
            )

    # --- apply the cut (identical in both modes) ---
    new_pass_reco_mask = np.ones(len(events["pass_reco"]), dtype=bool)
    SF_mask = np.ones(len(events["pass_reco"]), dtype=bool)
    for sector in range(num_sectors):
        sector_mask = electrons["sector"] == (sector + 1)

        popt_mu = popt_mu_by_sector[sector]
        popt_sigma = popt_sigma_by_sector[sector]
        edep_in_sector = electrons["total_ecal_energy"][sector_mask]
        sampling_fraction_in_sector = electrons["SF"][sector_mask]
        fit_mu = sf_fit_function(edep_in_sector, *popt_mu)
        fit_sigma = sf_fit_function(edep_in_sector, *popt_sigma)

        if log_file is not None:
            with open(log_file, "a") as f:
                f.write(f"\nSector {sector+1} sampling fraction fits\n")
                f.write("a_mu & b_mu & c_mu & a_sigma & b_sigma & c_sigma\n")
                row = (
                    f"{popt_mu[0]:.6f} & {popt_mu[1]:.6f} & {popt_mu[2]:.6f}"
                    f" & {popt_sigma[0]:.6f} & {popt_sigma[1]:.6f} & {popt_sigma[2]:.6f}"
                    " \\\\\n"
                )
                f.write(row)

        SF_mask[sector_mask] = (
            sampling_fraction_in_sector < (fit_mu + 3.5 * fit_sigma)
        ) & (sampling_fraction_in_sector > (fit_mu - 3.5 * fit_sigma))
        new_pass_reco_mask[sector_mask] = (
            (events["pass_reco"][sector_mask])
            & (sampling_fraction_in_sector < (fit_mu + 3.5 * fit_sigma))
            & (sampling_fraction_in_sector > (fit_mu - 3.5 * fit_sigma))
        )

    events["pass_reco"] = new_pass_reco_mask
    events["pass_SF"] = SF_mask
    print(f"Have {ak.sum(events['pass_reco'])} events after SF cuts")
    if log_file is not None:
        with open(log_file, "a") as f:
            f.write(
                f"\nHave {ak.sum(events['pass_reco'])} pass reco events after SF cuts\n"
            )
            f.write(
                f"Have {ak.sum(SF_mask)/number_of_initial_electrons} fraction of events after SF cuts\n"
            )
            f.write(
                f"Have {ak.sum((events['pass_fiducial']) & (SF_mask))/number_of_initial_electrons} fraction of events after fiducial and SF cuts\n"
            )
            f.write(
                f"Have {ak.sum((events['pass_partial_SF']) & (SF_mask))/number_of_initial_electrons} fraction of events after partial and SF cuts\n"
            )
            f.write(
                f"Have {ak.sum((events['pass_fiducial']) & (events['pass_partial_SF']) & (SF_mask))/ak.sum((events['pass_fiducial']))} fraction of events that pass fiducial cuts, but also partial and SF cuts\n"
            )
    return events


def double_gaussian(x, amp1, mean1, sigma1, amp2, mean2, sigma2):
    return amp1 * np.exp(-((x - mean1) ** 2) / (2 * sigma1)) + amp2 * np.exp(
        -((x - mean2) ** 2) / (2 * sigma2)
    )


def _fit_vertex_z_sectors(electrons, pass_reco, num_sectors):
    """Double-Gaussian fit of v_z per sector. ONLY called in develop_cuts mode."""
    z_fit_parameters_all_sectors = []
    for sector in range(num_sectors):
        sector_cut = (electrons["sector"] == (sector + 1)) & (pass_reco)
        vertex_z = np.array(electrons["v_z"][sector_cut])
        vertex_z_counts, vertex_z_bins = np.histogram(
            vertex_z, bins=100, range=(-12, 5)
        )
        vertex_z_bin_centers = (vertex_z_bins[:-1] + vertex_z_bins[1:]) / 2

        amplitude_guess = np.max(vertex_z_counts)

        z_fit_parameters, _ = curve_fit(
            double_gaussian,
            vertex_z_bin_centers,
            vertex_z_counts,
            p0=(amplitude_guess, -7, 2.5, amplitude_guess, -1.5, 1.5),
        )
        z_fit_parameters_all_sectors.append(z_fit_parameters)

    return z_fit_parameters_all_sectors


def _derive_target_params(z_fit_parameters_all_sectors, num_sectors):
    """Pure arithmetic (no fitting): assign which Gaussian is LD2 vs solid, per sector.
    ONLY called in develop_cuts mode -- result is what gets cached."""
    deuterium_mean_by_sector, deuterium_sigma_by_sector = [], []
    solid_mean_by_sector, solid_sigma_by_sector = [], []

    for sector in range(num_sectors):
        z_fit_parameters = z_fit_parameters_all_sectors[sector]
        deuterium_z_mean = min(z_fit_parameters[1], z_fit_parameters[4])
        if deuterium_z_mean == z_fit_parameters[1]:
            deuterium_z_sigma = z_fit_parameters[2]
            solid_z_mean, solid_z_sigma = z_fit_parameters[4], z_fit_parameters[5]
        else:
            deuterium_z_sigma = z_fit_parameters[5]
            solid_z_mean, solid_z_sigma = z_fit_parameters[1], z_fit_parameters[2]

        deuterium_mean_by_sector.append(float(deuterium_z_mean))
        deuterium_sigma_by_sector.append(float(deuterium_z_sigma))
        solid_mean_by_sector.append(float(solid_z_mean))
        solid_sigma_by_sector.append(float(solid_z_sigma))

    return (
        deuterium_mean_by_sector,
        deuterium_sigma_by_sector,
        solid_mean_by_sector,
        solid_sigma_by_sector,
    )


def _plot_zvertex_fits(
    electrons,
    pass_reco,
    z_fit_parameters_all_sectors,
    num_sectors,
    plots_directory,
    plot_title,
):
    """Raw fit diagnostic. Develop mode only -- nothing to show here in apply mode."""
    fig, axs = plt.subplots(3, 2, figsize=(18, 18))
    axs = axs.flatten()
    for sector in range(num_sectors):
        sector_cut = (electrons["sector"] == (sector + 1)) & (pass_reco)
        vertex_z = np.array(electrons["v_z"][sector_cut])
        vertex_z_counts, vertex_z_bins = np.histogram(
            vertex_z, bins=100, range=(-12, 5)
        )
        vertex_z_bin_centers = (vertex_z_bins[:-1] + vertex_z_bins[1:]) / 2
        y_fit = double_gaussian(
            vertex_z_bin_centers, *z_fit_parameters_all_sectors[sector]
        )

        axs[sector].hist(vertex_z, bins=100, range=(-12, 5), histtype="step")
        axs[sector].plot(vertex_z_bin_centers, y_fit, color="r")
        axs[sector].set_xlabel("$v_{z}$ (cm)")
        axs[sector].set_title(f"Sector {sector+1}")

    fig.tight_layout()
    if plot_title is not None:
        plt.suptitle(plot_title, y=1.0)
    if plots_directory is not None:
        plt.savefig(plots_directory + "zvertex_fits.png")
    plt.close()


def _plot_target_selections(
    electrons,
    pass_reco,
    deuterium_mean_by_sector,
    deuterium_sigma_by_sector,
    solid_mean_by_sector,
    solid_sigma_by_sector,
    solid_target_name,
    num_sectors,
    plots_directory,
    plot_title,
):
    """Application diagnostic -- runs in BOTH modes, purely from cached numbers."""
    fig, axs = plt.subplots(3, 2, figsize=(18, 18))
    axs = axs.flatten()

    for sector in range(num_sectors):
        sector_cut = (electrons["sector"] == (sector + 1)) & (pass_reco)
        vertex_z = np.array(electrons["v_z"][sector_cut])

        deuterium_z_mean, deuterium_z_sigma = (
            deuterium_mean_by_sector[sector],
            deuterium_sigma_by_sector[sector],
        )
        solid_z_mean, solid_z_sigma = (
            solid_mean_by_sector[sector],
            solid_sigma_by_sector[sector],
        )

        deuterium_cut = (vertex_z > (deuterium_z_mean - 3 * deuterium_z_sigma)) & (
            vertex_z < (deuterium_z_mean + 3 * deuterium_z_sigma)
        )
        solid_cut = (vertex_z > (solid_z_mean - 5 * solid_z_sigma)) & (
            vertex_z < (solid_z_mean + 5 * solid_z_sigma)
        )

        axs[sector].hist(vertex_z, bins=100, range=(-12, 5), histtype="step")
        axs[sector].hist(
            vertex_z[deuterium_cut],
            bins=100,
            range=(-12, 5),
            color="b",
            label="LD2",
            alpha=0.8,
        )
        axs[sector].hist(
            vertex_z[solid_cut],
            bins=100,
            range=(-12, 5),
            color="r",
            label=solid_target_name,
            alpha=0.8,
        )
        axs[sector].set_xlabel("$v_{z}$ (cm)")
        axs[sector].set_title(f"Sector {sector+1}")
        axs[sector].legend(loc="upper left")

    fig.tight_layout()
    if plot_title is not None:
        plt.suptitle(plot_title, y=1.0)
    if plots_directory is not None:
        plt.savefig(plots_directory + "target_selections.png")
    plt.close()


"""
Selecting LD2 vs. solid target using a double Gaussian fit
There is a develop_cuts parameter that when True, the fits are done and fit values store in a json
If this is false (default) the fit values are loaded from the json
"""


def apply_target_selection(
    events,
    solid_target_name,
    develop_cuts=False,
    cut_params_path=None,
    save_plots=True,
    plots_directory=None,
    plot_title=None,
    log_file=None,
    number_of_initial_electrons=None,
):
    num_sectors = 6
    electrons = events["reconstructed"]

    if develop_cuts:
        z_fit_parameters_all_sectors = _fit_vertex_z_sectors(
            electrons, events["pass_reco"], num_sectors
        )
        (
            deuterium_mean_by_sector,
            deuterium_sigma_by_sector,
            solid_mean_by_sector,
            solid_sigma_by_sector,
        ) = _derive_target_params(z_fit_parameters_all_sectors, num_sectors)

        if save_plots:
            _plot_zvertex_fits(
                electrons,
                events["pass_reco"],
                z_fit_parameters_all_sectors,
                num_sectors,
                plots_directory,
                plot_title,
            )
            _plot_target_selections(
                electrons,
                events["pass_reco"],
                deuterium_mean_by_sector,
                deuterium_sigma_by_sector,
                solid_mean_by_sector,
                solid_sigma_by_sector,
                solid_target_name,
                num_sectors,
                plots_directory,
                plot_title,
            )

        cut_params = {
            "deuterium_mean": deuterium_mean_by_sector,
            "deuterium_sigma": deuterium_sigma_by_sector,
            "solid_mean": solid_mean_by_sector,
            "solid_sigma": solid_sigma_by_sector,
        }
        if cut_params_path is not None:
            with open(cut_params_path, "w") as f:
                json.dump(cut_params, f, indent=2)

    else:
        if cut_params_path is None:
            raise ValueError("cut_params_path is required when develop_cuts=False")
        with open(cut_params_path, "r") as f:
            cut_params = json.load(f)

        deuterium_mean_by_sector = cut_params["deuterium_mean"]
        deuterium_sigma_by_sector = cut_params["deuterium_sigma"]
        solid_mean_by_sector = cut_params["solid_mean"]
        solid_sigma_by_sector = cut_params["solid_sigma"]

        if save_plots and plots_directory is not None:
            _plot_target_selections(
                electrons,
                events["pass_reco"],
                deuterium_mean_by_sector,
                deuterium_sigma_by_sector,
                solid_mean_by_sector,
                solid_sigma_by_sector,
                solid_target_name,
                num_sectors,
                plots_directory,
                plot_title,
            )

    # --- apply the cut (identical in both modes, no fitting/derivation, just lookup + mask) ---
    deuterium_mask = np.ones(len(events["pass_reco"]), dtype=bool)
    solid_mask = np.ones(len(events["pass_reco"]), dtype=bool)
    for sector in range(num_sectors):
        sector_mask = electrons["sector"] == (sector + 1)
        vertex_z_in_sector = electrons["v_z"][sector_mask]

        deuterium_mask_in_sector = (
            vertex_z_in_sector
            > (deuterium_mean_by_sector[sector] - 3 * deuterium_sigma_by_sector[sector])
        ) & (
            vertex_z_in_sector
            < (deuterium_mean_by_sector[sector] + 3 * deuterium_sigma_by_sector[sector])
        )
        solid_mask_in_sector = (
            vertex_z_in_sector
            > (solid_mean_by_sector[sector] - 5 * solid_sigma_by_sector[sector])
        ) & (
            vertex_z_in_sector
            < (solid_mean_by_sector[sector] + 5 * solid_sigma_by_sector[sector])
        )

        deuterium_mask[sector_mask] = (events["pass_reco"][sector_mask]) & (
            deuterium_mask_in_sector
        )
        solid_mask[sector_mask] = (events["pass_reco"][sector_mask]) & (
            solid_mask_in_sector
        )

        if log_file is not None:
            with open(log_file, "a") as f:
                f.write(f"\nSector {sector+1} target selection parameters:\n")
                f.write(
                    f"Deuterium mean (cm): {deuterium_mean_by_sector[sector]}, sigma (cm): {deuterium_sigma_by_sector[sector]}\n"
                )
                f.write(
                    f"{solid_target_name} mean (cm): {solid_mean_by_sector[sector]}, sigma (cm): {solid_sigma_by_sector[sector]}\n"
                )
                f.write("LD2 vz range (cm) | solid vz range (cm)")
                f.write(
                    f"${deuterium_mean_by_sector[sector] - 3*deuterium_sigma_by_sector[sector]} < v_z < {deuterium_mean_by_sector[sector] + 3*deuterium_sigma_by_sector[sector]}$ & ${solid_mean_by_sector[sector] - 5*solid_sigma_by_sector[sector]} < v_z < {solid_mean_by_sector[sector] + 5*solid_sigma_by_sector[sector]}$\n"
                )

    if log_file is not None:
        with open(log_file, "a") as f:
            f.write(
                f"Have {ak.sum((deuterium_mask) | (solid_mask))/ak.sum(events['pass_reco'])} fraction of events after target selection cuts\n"
            )

    events["pass_reco"] = (deuterium_mask) | (solid_mask)
    target = np.empty(len(events), dtype=object)
    target[deuterium_mask] = "LD2"
    target[solid_mask] = solid_target_name
    target[(~deuterium_mask) & (~solid_mask)] = "None"

    events["reconstructed"] = ak.with_field(
        events["reconstructed"], target.tolist(), "target"
    )

    print(f"Have {ak.sum(events['pass_reco'])} events after target selection cuts")
    if log_file is not None:
        with open(log_file, "a") as f:
            f.write(
                f"Have {ak.sum(events['pass_reco'])} events after target selection cuts\n"
            )

    return events
