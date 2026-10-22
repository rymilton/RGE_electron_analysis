# This file make plots of the cuts made by the Event Builder
# The following plots are needed:
# 1. PCAL energy cut
# 2. HTCC photoelectron cut
# 3. SF vs Edep fit plot
# The starting point of this file should be charged tracks for all events, and separately for only events with a trigger electron
# (since events without one are dropped later anyway). Each set of plots is saved in its own subdirectory.
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
    run_numbers = np.unique(ak.to_numpy(events["meta_info"]["run_number"]))
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


def plot_PCAL_energy(tracks, plot_title, plots_directory):
    sector = ak.to_numpy(ak.flatten(tracks["sector"]))
    E_PCAL = ak.to_numpy(ak.flatten(tracks["E_PCAL"]))
    E_EC = ak.to_numpy(ak.flatten(tracks["E_ECIN"] + tracks["E_ECOUT"]))

    fig, axs = plt.subplots(2, 3, figsize=(30, 18))
    for i, ax in enumerate(axs.flatten()):
        sector_mask = sector == i + 1
        _, _, _, mesh = ax.hist2d(
            E_PCAL[sector_mask],
            E_EC[sector_mask],
            bins=200,
            range=[[0, 1.5], [0, 1.5]],
            norm=colors.LogNorm(),
        )
        ax.axvline(0.06, color="red", label="$E_{PCAL}$ > 0.06 GeV cut")
        ax.legend()
        ax.set_xlabel("$E_{PCAL}$ (GeV)")
        ax.set_ylabel("$E_{ECIN} + E_{ECOUT}$ (GeV)")
        ax.set_title(f"Sector {i + 1}")

        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(mesh, cax=cax)

    plt.suptitle(plot_title)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_directory, "PCAL_energy.png"))
    plt.close()


def plot_HTCC_photoelectrons(tracks, plot_title, plots_directory):
    sector = ak.to_numpy(ak.flatten(tracks["sector"]))
    Nphe_HTCC = ak.to_numpy(ak.flatten(tracks["Nphe_HTCC"]))
    # Tracks without an HTCC entry are -9999
    has_HTCC = Nphe_HTCC != -9999

    fig, ax = plt.subplots(figsize=(12, 10))
    for i in range(1, 7):
        ax.hist(
            Nphe_HTCC[(sector == i) & has_HTCC],
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

    for subdirectory, require_trigger_electron in [
        ("all_events", False),
        ("trigger_electron_events", True),
    ]:
        negative_tracks = get_negative_tracks(events, require_trigger_electron)
        print(
            f"{subdirectory}: selected {ak.sum(ak.num(negative_tracks['track_charge']))} "
            f"negative tracks from {len(negative_tracks)} events"
        )

        plots_directory = os.path.join(flags.plots_directory, subdirectory)
        os.makedirs(plots_directory, exist_ok=True)
        plot_PCAL_energy(negative_tracks, plot_title, plots_directory)
        plot_HTCC_photoelectrons(negative_tracks, plot_title, plots_directory)


if __name__ == "__main__":
    main()
