import argparse
import uproot
import glob
import awkward as ak
import numpy as np
import time
import h5py as h5
import os
from utils import LoadYaml, open_data
from selection_functions import *


def parse_arguments():
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        "--input_file",
        default="/home/rmilton/work_dir/rge_datasets/job_9586_LD2Csolid_clasdis_deuteron_zh0_3k/eventbuilder_electrons/electrons_eventbuilder_LD2Csolid_clasdis_deuteron_100mil_zh0-9586-0.root",
        help="ROOT file containing event builder electrons after running eventbuilder_electron_selection.py",
        type=str,
    )
    parser.add_argument(
        "--output_directory",
        default="/home/rmilton/work_dir/rge_datasets/job_9586_LD2Csolid_clasdis_deuteron_zh0_3k/candidate_electrons/",
        help="Directory to candidate electrons",
        type=str,
    )
    parser.add_argument(
        "--output_file",
        default="candidate_electrons_LD2Csolid_clasdis_deuteron_100mil_zh0-9586-0.root",
        help="Name of output ROOT file",
        type=str,
    )
    parser.add_argument(
        "--save_MC",
        action="store_true",
        default=False,
        help="Load Monte Carlo information from file and save it in output electrons",
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
        "--plots_directory",
        default="./plots/",
        help="Directory to store plots",
        type=str,
    )
    parser.add_argument(
        "--save_plots",
        action="store_true",
        default=False,
        help="Save the plots that are generated during the analysis",
    )
    parser.add_argument(
        "--simulation",
        action="store_true",
        default=False,
        help="Use this flag if you're using simulated data rather than actual data",
    )


    flags = parser.parse_args()

    return flags


def main():
    flags = parse_arguments()
    
    # Open data
    print(flags.config, flags.config_directory)
    parameters = LoadYaml(flags.config, flags.config_directory)

    events_array = open_data(
        data_path = flags.input_file,
        branches_to_open = parameters["BRANCHES_TO_SAVE"],
        data_tree_name = "reconstructed_electrons",
        open_MC = flags.save_MC,
        MC_branches_to_open = parameters["MC_BRANCHES_TO_SAVE"] if flags.save_MC else None,
        MC_tree_name = "MC_electrons",
    )
    # Apply DIS cuts and other basic cuts
    events_array = apply_kinematic_cuts(events_array, parameters["ELECTRON_KINEMATIC_CUTS"])
    # Apply fiducial cuts
    if flags.save_plots:
        os.makedirs(flags.plots_directory, exist_ok=True)
        plot_title = "RGE LD2 + C: clasdis simulation solid"
    else:
        plot_title = None
    events_array = apply_fiducial_cuts(
        events = events_array,
        fiducial_cuts = parameters["ELECTRON_FIDUCIAL_CUTS"],
        save_plots = flags.save_plots,
        plots_directory = flags.plots_directory,
        plot_title = plot_title,
    )
    
    # Apply partial sampling fraction cuts
    events_array = apply_partial_sampling_fraction_cut(
        events = events_array,
        is_simulation = flags.simulation,
        save_plots = flags.save_plots,
        plots_directory = flags.plots_directory,
        plot_title = plot_title,
    )
    # Apply SF cuts
    
    # Save the cut electrons. Should have the option to cut on targets or not
    
if __name__ == "__main__":
    main()