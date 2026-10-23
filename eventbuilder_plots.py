# This file make plots of the cuts made by the Event Builder
# The following plots are needed:
# 1. PCAL energy cut
# 2. HTCC photoelectron cut
# 3. SF vs Edep fit plot
# The starting point of this file should be charged tracks for all events, and separately for only events with a trigger electron
# (since events without one are dropped later anyway), for tracks with energy in both the PCAL and ECIN + ECOUT
# (in all events and in events with a trigger electron), and for only EB electrons (pid == 11).
# Each set of plots is saved in its own subdirectory.
# The input files should be those coming from the tuple maker
# In these plots, use all negative tracks.
# Similar to the other scripts, this should be done either for an array of input files or a single file.

import argparse
import os
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from mpl_toolkits.axes_grid1 import make_axes_locatable
import awkward as ak
import numpy as np
import mplhep as hep
from utils import LoadYaml, open_data

hep.style.use("CMS")

# Event builder sampling fraction cut parameters for each sector, used in
# sf_mean() and sf_sigma()
SF_CUT_PARAMETERS = {
    1: {
        "sf1": 0.23372,
        "sf2": 1.0,
        "sf3": -0.01726,
        "sf4": 0.00050,
        "sfs1": 0.01763,
        "sfs2": 1.0,
        "sfs3": 0.0,
        "sfs4": 0.0,
    },
    2: {
        "sf1": 0.24209,
        "sf2": 1.0,
        "sf3": -0.03649,
        "sf4": 0.00063,
        "sfs1": 0.02018,
        "sfs2": 1.0,
        "sfs3": 0.0,
        "sfs4": 0.0,
    },
    3: {
        "sf1": 0.23375,
        "sf2": 1.0,
        "sf3": -0.01600,
        "sf4": 0.00070,
        "sfs1": 0.01719,
        "sfs2": 1.0,
        "sfs3": 0.0,
        "sfs4": 0.0,
    },
    4: {
        "sf1": 0.25006,
        "sf2": 1.0,
        "sf3": -0.01613,
        "sf4": 0.00050,
        "sfs1": 0.01687,
        "sfs2": 1.0,
        "sfs3": 0.0,
        "sfs4": 0.0,
    },
    5: {
        "sf1": 0.23677,
        "sf2": 1.0,
        "sf3": -0.01963,
        "sf4": 0.00050,
        "sfs1": 0.01744,
        "sfs2": 1.0,
        "sfs3": 0.0,
        "sfs4": 0.0,
    },
    6: {
        "sf1": 0.23757,
        "sf2": 1.0,
        "sf3": -0.01600,
        "sf4": 0.00070,
        "sfs1": 0.01585,
        "sfs2": 1.0,
        "sfs3": 0.0,
        "sfs4": 0.0,
    },
}

# Branches from tuple_maker.cpp's "reconstructed" tree. All are per-track (multiple tracks per event, i.e. 2D arrays)
# except has_trigger_electron, which is per-event
BRANCHES_TO_OPEN = [
    "has_trigger_electron",
    "track_charge",
    "p",
    "sector",
    "E_PCAL",
    "E_ECIN",
    "E_ECOUT",
    "Nphe_HTCC",
    "pid",
]


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Make plots of the cuts imposed by the CLAS12 Event Builder"
    )
    parser.add_argument(
        "--input_file",
        default=None,
        help="ROOT file from tuple_maker.cpp (ntuples_*.root)",
        type=str,
    )
    parser.add_argument(
        "--input_file_array",
        nargs="+",  # one or more values
        default=None,
        help="Multiple ROOT files from tuple_maker.cpp. This will be chosen over input_file if given",
        type=str,
    )
    parser.add_argument(
        "--nmax",
        default=None,
        help="Max number of events to load",
        type=int,
    )
    parser.add_argument(
        "--num_processes",
        default=4,
        help="Number of processes to use when opening multiple input files in parallel",
        type=int,
    )
    parser.add_argument(
        "--plots_directory",
        default="./eventbuilder_plots/",
        help="Directory to store plots",
        type=str,
    )
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Basic config file containing general options",
        type=str,
    )
    parser.add_argument(
        "--config_directory",
        default="./configs/",
        help="Directory containing the config files",
        type=str,
    )
    parser.add_argument(
        "--simulation",
        action="store_true",
        default=False,
        help="Use this flag if you're using simulated data rather than actual data",
    )
    parser.add_argument(
        "--data_name",
        default=None,
        help="Name of the simulation (e.g., clasdis_solid, GiBUU_liquid, etc.) in PLOT_TITLES. Only used with --simulation",
        type=str,
    )
    parser.add_argument(
        "--solid_target",
        default="C",
        help="Name of solid target",
        type=str,
    )
    return parser.parse_args()


def get_input_files(flags):
    if flags.input_file_array is not None:
        return flags.input_file_array
    return [flags.input_file]


def get_plot_title(flags, parameters, events):
    if flags.simulation:
        return f"RG-E LD2 + {flags.solid_target} : {parameters['PLOT_TITLES'][flags.data_name]}"
    # Events without a RUN::config bank have run_number <= 0
    run_numbers = np.unique(events["meta_info"]["run_number"])
    run_numbers = run_numbers[run_numbers > 0]
    if len(run_numbers) == 1:
        run_label = f"{run_numbers[0]}"
    else:
        run_label = f"Runs {run_numbers[0]}-{run_numbers[-1]}"
    return f"RG-E LD2 + {flags.solid_target} : {run_label} Pass 1"


def get_negative_tracks(events, require_trigger_electron):
    # Selects all negatively charged tracks within each event, optionally only
    # for events with a trigger electron. has_trigger_electron is removed after
    # the event selection since it's per-event and can't be masked per-track.
    tracks = events["reconstructed"]
    if require_trigger_electron:
        tracks = tracks[tracks["has_trigger_electron"]]
    tracks = ak.without_field(tracks, "has_trigger_electron")
    return tracks[tracks["track_charge"] < 0]


def select_nonzero_pcal_ecal_energy(tracks):
    # Keeps tracks with energy in both the PCAL and ECIN + ECOUT
    return tracks[
        (tracks["E_PCAL"] > 0) & (tracks["E_ECIN"] + tracks["E_ECOUT"] > 0)
    ]


def plot_PCAL_energy(
    tracks, plot_title, plots_directory, apply_cuts=False, max_energy=0.6
):
    # apply_cuts=True keeps only tracks passing the HTCC (Nphe > 2) and SF
    # (within mu +- 5 sigma) cuts. max_energy sets the upper edge of both axes (GeV).
    sector = ak.to_numpy(ak.flatten(tracks["sector"]))
    E_PCAL = ak.to_numpy(ak.flatten(tracks["E_PCAL"]))
    E_EC = ak.to_numpy(ak.flatten(tracks["E_ECIN"] + tracks["E_ECOUT"]))
    Nphe_HTCC = ak.to_numpy(ak.flatten(tracks["Nphe_HTCC"]))
    E_dep = E_PCAL + E_EC
    SF = E_dep / ak.to_numpy(ak.flatten(tracks["p"]))

    fig, axs = plt.subplots(2, 3, figsize=(30, 18))
    for i, ax in enumerate(axs.flatten()):
        # Tracks with no calorimeter energy at all would pile up in the (0, 0) bin
        sector_mask = (sector == i + 1) & (E_dep > 0)
        if apply_cuts:
            params = SF_CUT_PARAMETERS[i + 1]
            # Tracks with E_dep = 0 give inf/nan for mu and sigma and fail the SF cut
            with np.errstate(divide="ignore", invalid="ignore"):
                mu = sf_mean(E_dep, params)
                sigma = sf_sigma(E_dep, params)
            sector_mask = (
                (sector_mask)
                & (Nphe_HTCC > 2)
                & (np.abs(SF - mu) < 5 * sigma)
            )
        _, _, _, mesh = ax.hist2d(
            E_PCAL[sector_mask],
            E_EC[sector_mask],
            bins=200,
            range=[[0, max_energy], [0, max_energy]],
            norm=colors.LogNorm(),
        )
        ax.axvline(0.06, color="red", label="$E_{PCAL}$ > 0.06 GeV cut")
        ax.legend(frameon=True, facecolor="white", edgecolor="black", framealpha=1)
        ax.set_xlabel("$E_{PCAL}$ (GeV)")
        ax.set_ylabel("$E_{ECIN} + E_{ECOUT}$ (GeV)")
        ax.set_title(f"Sector {i + 1}")

        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(mesh, cax=cax)

    plt.suptitle(plot_title)
    plt.tight_layout()
    file_name = "PCAL_energy_HTCC_SF_cuts" if apply_cuts else "PCAL_energy"
    if max_energy != 0.6:
        file_name += f"_max_{max_energy}_GeV"
    plt.savefig(os.path.join(plots_directory, file_name + ".png"))
    plt.close()


def plot_PCAL_energy_by_sector(tracks, plot_title, plots_directory):
    # One figure per sector: PCAL vs. ECIN + ECOUT energy for all negative
    # tracks (left) and for those passing the HTCC and SF cuts (right)
    sector = ak.to_numpy(ak.flatten(tracks["sector"]))
    E_PCAL = ak.to_numpy(ak.flatten(tracks["E_PCAL"]))
    E_EC = ak.to_numpy(ak.flatten(tracks["E_ECIN"] + tracks["E_ECOUT"]))
    Nphe_HTCC = ak.to_numpy(ak.flatten(tracks["Nphe_HTCC"]))
    E_dep = E_PCAL + E_EC
    SF = E_dep / ak.to_numpy(ak.flatten(tracks["p"]))

    for i in range(1, 7):
        # Tracks with no calorimeter energy at all would pile up in the (0, 0) bin
        sector_mask = (sector == i) & (E_dep > 0)
        params = SF_CUT_PARAMETERS[i]
        # Tracks with E_dep = 0 give inf/nan for mu and sigma and fail the SF cut
        with np.errstate(divide="ignore", invalid="ignore"):
            mu = sf_mean(E_dep, params)
            sigma = sf_sigma(E_dep, params)
        pass_cuts = (Nphe_HTCC > 2) & (np.abs(SF - mu) < 5 * sigma)

        fig, axs = plt.subplots(1, 2, figsize=(24, 10))
        for ax, mask, title in [
            (axs[0], sector_mask, "All negative tracks"),
            (axs[1], (sector_mask) & (pass_cuts), "Pass $N_{phe}$ + SF cuts"),
        ]:
            _, _, _, mesh = ax.hist2d(
                E_PCAL[mask],
                E_EC[mask],
                bins=200,
                range=[[0, 0.6], [0, 0.6]],
                norm=colors.LogNorm(),
            )
            ax.axvline(0.06, color="red", label="$E_{PCAL}$ > 0.06 GeV cut")
            ax.legend(frameon=True, facecolor="white", edgecolor="black", framealpha=1)
            ax.set_xlabel("$E_{PCAL}$ (GeV)")
            ax.set_ylabel("$E_{ECIN} + E_{ECOUT}$ (GeV)")
            ax.set_title(f"Sector {i} - {title}")

            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            fig.colorbar(mesh, cax=cax)

        plt.suptitle(plot_title)
        plt.tight_layout()
        plt.savefig(os.path.join(plots_directory, f"PCAL_energy_cuts_sector{i}.png"))
        plt.close()


def plot_HTCC_photoelectrons(tracks, plot_title, plots_directory):
    sector = ak.flatten(tracks["sector"])
    Nphe_HTCC = ak.flatten(tracks["Nphe_HTCC"])
    # Tracks without an HTCC entry are -9999
    has_HTCC = Nphe_HTCC != -9999

    fig, ax = plt.subplots(figsize=(12, 10))
    for i in range(1, 7):
        ax.hist(
            Nphe_HTCC[(sector == i) & (has_HTCC)],
            bins=np.arange(0, 76),
            histtype="step",
            linewidth=2,
            density=True,
            label=f"Sector {i}",
        )
    ax.axvline(2, color="red", label="$N_{phe}$ > 2 cut")
    ax.legend()
    ax.set_xlabel("HTCC $N_{phe}$")
    ax.set_ylabel("Density")
    ax.set_title(plot_title)

    plt.tight_layout()
    plt.savefig(os.path.join(plots_directory, "HTCC_photoelectrons.png"))
    plt.close()


def sf_mean(E_dep, params):
    return params["sf1"] * (
        params["sf2"] + params["sf3"] / E_dep + params["sf4"] / E_dep**2
    )


def sf_sigma(E_dep, params):
    return params["sfs1"] * (
        params["sfs2"] + params["sfs3"] / E_dep + params["sfs4"] / E_dep**2
    )


def plot_sampling_fraction(tracks, plot_title, plots_directory, apply_cuts=False):
    # apply_cuts=True keeps only tracks passing the HTCC (Nphe > 2) and PCAL
    # energy (E_PCAL > 0.06 GeV) cuts
    sector = ak.to_numpy(ak.flatten(tracks["sector"]))
    E_PCAL = ak.to_numpy(ak.flatten(tracks["E_PCAL"]))
    E_dep = E_PCAL + ak.to_numpy(ak.flatten(tracks["E_ECIN"] + tracks["E_ECOUT"]))
    SF = E_dep / ak.to_numpy(ak.flatten(tracks["p"]))
    Nphe_HTCC = ak.to_numpy(ak.flatten(tracks["Nphe_HTCC"]))
    pass_cuts = (Nphe_HTCC > 2) & (E_PCAL > 0.06) if apply_cuts else True

    fig, axs = plt.subplots(2, 3, figsize=(30, 18))
    for i, ax in enumerate(axs.flatten()):
        # Tracks with no calorimeter energy at all would pile up in the (0, 0) bin
        sector_mask = (sector == i + 1) & (pass_cuts) & (E_dep > 0)
        _, _, _, mesh = ax.hist2d(
            E_dep[sector_mask],
            SF[sector_mask],
            bins=200,
            range=[[0, 2.5], [0, 0.4]],
            norm=colors.LogNorm(),
        )

        # Starting above 0 GeV since mu and sigma diverge as 1/E_dep^2
        E_dep_curve = np.linspace(0.000001, 2.5, 500)
        params = SF_CUT_PARAMETERS[i + 1]
        mu = sf_mean(E_dep_curve, params)
        sigma = sf_sigma(E_dep_curve, params)
        ax.plot(E_dep_curve, mu, color="red", linewidth=2, label="Mean SF")
        ax.plot(
            E_dep_curve,
            mu + 5 * sigma,
            color="black",
            linewidth=2,
            label=r"$\pm 5\sigma$",
        )
        ax.plot(E_dep_curve, mu - 5 * sigma, color="black", linewidth=2)
        ax.set_xlim(0, 2.5)
        ax.set_ylim(0, 0.4)
        ax.legend(
            loc="upper right",
            frameon=True,
            facecolor="white",
            edgecolor="black",
            framealpha=1,
        )

        ax.set_xlabel("$E_{dep, total}$ (GeV)")
        ax.set_ylabel("$E_{dep, total}/p$")
        ax.set_title(f"Sector {i + 1}")

        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(mesh, cax=cax)

    plt.suptitle(plot_title)
    plt.tight_layout()
    file_name = (
        "sampling_fraction_HTCC_PCAL_cuts.png"
        if apply_cuts
        else "sampling_fraction.png"
    )
    plt.savefig(os.path.join(plots_directory, file_name))
    plt.close()


def write_electron_summary(tracks, plots_directory):
    # Counts EB electrons (pid == 11) among negative FD tracks and checks that the
    # PCAL, HTCC, and SF cuts reproduce the EB electron ID. Electrons failing any
    # cut mean the cut values (e.g. SF_CUT_PARAMETERS) don't match this run.
    tracks = tracks[tracks["sector"] > 0]
    sector = ak.flatten(tracks["sector"])
    E_PCAL = ak.flatten(tracks["E_PCAL"])
    E_dep = E_PCAL + ak.flatten(tracks["E_ECIN"] + tracks["E_ECOUT"])
    SF = E_dep / ak.flatten(tracks["p"])
    Nphe_HTCC = ak.flatten(tracks["Nphe_HTCC"])
    is_electron = ak.flatten(tracks["pid"]) == 11

    pass_PCAL = E_PCAL > 0.06
    pass_HTCC = Nphe_HTCC > 2
    pass_SF = False
    for i in range(1, 7):
        # Tracks with E_dep = 0 give inf/nan for mu and sigma and fail the SF cut
        with np.errstate(divide="ignore", invalid="ignore"):
            mu = sf_mean(E_dep, SF_CUT_PARAMETERS[i])
            sigma = sf_sigma(E_dep, SF_CUT_PARAMETERS[i])
        pass_SF = (pass_SF) | ((sector == i) & (np.abs(SF - mu) < 5 * sigma))
    pass_all = (pass_PCAL) & (pass_HTCC) & (pass_SF)
    fail_any = (is_electron) & (~pass_all)

    lines = [
        f"Negative FD tracks: {len(is_electron)}",
        f"EB electrons (pid == 11): {np.sum(is_electron)} ({100 * np.mean(is_electron):.1f}%)",
        "Cuts (E_PCAL > 0.06 GeV, Nphe_HTCC > 2, SF within mu +- 5 sigma) vs. pid == 11:",
        f"  Pass all cuts, pid == 11: {np.sum((pass_all) & (is_electron))}",
        f"  Pass all cuts, pid != 11: {np.sum((pass_all) & (~is_electron))}",
        f"  pid == 11, fail any cut: {np.sum(fail_any)} "
        f"(PCAL: {np.sum((fail_any) & (~pass_PCAL))}, HTCC: {np.sum((fail_any) & (~pass_HTCC))}, "
        f"SF: {np.sum((fail_any) & (~pass_SF))})",
    ]
    summary = "\n".join(lines)
    print(summary)
    with open(os.path.join(plots_directory, "summary.txt"), "w") as f:
        f.write(summary + "\n")


def plot_track_multiplicity(events, plot_title, plots_directory):
    # Number of total, positive, and negative tracks (REC::Track, FD and CD) per event
    track_charge = events["reconstructed"]["track_charge"]

    fig, ax = plt.subplots(figsize=(12, 10))
    for name, num_tracks in [
        ("Total", ak.num(track_charge)),
        ("Positive", ak.sum(track_charge > 0, axis=1)),
        ("Negative", ak.sum(track_charge < 0, axis=1)),
    ]:
        ax.hist(
            num_tracks,
            bins=np.arange(-0.5, np.max(num_tracks) + 1.5),
            histtype="step",
            linewidth=2,
            label=f"{name}: mean = {np.mean(num_tracks):.2f}, std. dev. = {np.std(num_tracks):.2f}",
        )
    ax.set_yscale("log")
    ax.legend()
    ax.set_xlabel("Number of tracks per event")
    ax.set_ylabel("Events")
    ax.set_title(plot_title)

    plt.tight_layout()
    plt.savefig(os.path.join(plots_directory, "track_multiplicity.png"))
    plt.close()


def main():
    flags = parse_arguments()
    parameters = LoadYaml(flags.config, flags.config_directory)

    # tuple_maker.cpp only writes the "meta" tree (with run_number) for real data
    events = open_data(
        get_input_files(flags),
        BRANCHES_TO_OPEN,
        data_tree_name="reconstructed",
        nmax=flags.nmax,
        get_meta_info=not flags.simulation,
        meta_tree_name="meta",
        num_processes=flags.num_processes,
    )
    plot_title = get_plot_title(flags, parameters, events)

    all_tracks = get_negative_tracks(events, require_trigger_electron=False)
    os.makedirs(flags.plots_directory, exist_ok=True)
    write_electron_summary(all_tracks, flags.plots_directory)
    plot_track_multiplicity(events, plot_title, flags.plots_directory)

    # Each set is built only when its plots are made, to avoid holding all of them in memory at once
    track_sets = [
        ("all_events", lambda: all_tracks),
        (
            "trigger_electron_events",
            lambda: get_negative_tracks(events, require_trigger_electron=True),
        ),
        ("nonzero_pcal_ecal_energy", lambda: select_nonzero_pcal_ecal_energy(all_tracks)),
        (
            "trigger_electron_nonzero_pcal_ecal_energy",
            lambda: select_nonzero_pcal_ecal_energy(
                get_negative_tracks(events, require_trigger_electron=True)
            ),
        ),
        ("electrons", lambda: all_tracks[all_tracks["pid"] == 11]),
    ]

    for subdirectory, get_tracks in track_sets:
        negative_tracks = get_tracks()
        print(
            f"{subdirectory}: selected {ak.sum(ak.num(negative_tracks['track_charge']))} "
            f"negative tracks from {len(negative_tracks)} events"
        )

        plots_directory = os.path.join(flags.plots_directory, subdirectory)
        os.makedirs(plots_directory, exist_ok=True)
        for max_energy in [0.6, 1.5]:
            plot_PCAL_energy(
                negative_tracks, plot_title, plots_directory, max_energy=max_energy
            )
            plot_PCAL_energy(
                negative_tracks,
                plot_title,
                plots_directory,
                apply_cuts=True,
                max_energy=max_energy,
            )
        plot_PCAL_energy_by_sector(negative_tracks, plot_title, plots_directory)
        plot_HTCC_photoelectrons(negative_tracks, plot_title, plots_directory)
        plot_sampling_fraction(negative_tracks, plot_title, plots_directory)
        plot_sampling_fraction(
            negative_tracks, plot_title, plots_directory, apply_cuts=True
        )


if __name__ == "__main__":
    main()
