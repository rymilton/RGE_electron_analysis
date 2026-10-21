# This file make plots of the cuts made by the Event Builder
# The following plots are needed:
# 1. PCAL energy cut
# 2. HTCC photoelectron cut
# 3. SF vs Edep fit plot
# The starting point of this file should be charged tracks for all events, even those without a trigger electron.
# The input files should be those coming from the tuple maker
# In these plots, use all negative tracks.
# Similar to the other scripts, this should be done either for an array of input files or a single file.

import argparse
import matplotlib.pyplot as plt
import awkward as ak
import numpy as np
import mplhep as hep
from utils import open_data

hep.style.use("CMS")

# Per-track branches from tuple_maker.cpp's "reconstructed" tree (multiple tracks per event, i.e. 2D arrays)
BRANCHES_TO_OPEN = [
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
        default="./plots/eventbuilder/",
        help="Directory to store plots",
        type=str,
    )
    return parser.parse_args()


def get_input_files(flags):
    if flags.input_file_array is not None:
        return flags.input_file_array
    return [flags.input_file]


def get_negative_tracks(events):
    # Keeps every event (including those without a trigger electron) and
    # selects all negatively charged tracks within each event.
    tracks = events["reconstructed"]
    return tracks[tracks["track_charge"] < 0]


def main():
    flags = parse_arguments()

    events = open_data(
        get_input_files(flags),
        BRANCHES_TO_OPEN,
        data_tree_name="reconstructed",
        nmax=flags.nmax,
        num_processes=flags.num_processes,
    )
    negative_tracks = get_negative_tracks(events)
    print(
        f"Selected {ak.sum(ak.num(negative_tracks['track_charge']))} negative tracks "
        f"from {len(negative_tracks)} events"
    )


if __name__ == "__main__":
    main()
