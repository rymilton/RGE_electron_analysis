import argparse
import uproot
import glob
import awkward as ak
import numpy as np
import time
import h5py as h5
import os
from utils import LoadYaml, open_data, save_output, CSV_to_df
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
        "--input_file_array",
        nargs="+",  # one or more values
        default=None,
        help="Multiple ROOT files containing event builder electrons after running eventbuilder_electron_selection.py. This will be chosen over input_file if given",
        type=str,
    )
    parser.add_argument(
        "--nmax",
        default=None,
        help="Max number of events to load",
        type=int,
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
    parser.add_argument(
        "--data_name",
        default="data_C",
        help="Name of the data (data_C) or simulation (e.g., clasdis_solid, GiBUU_liquid, etc.)",
        type=str,
    )
    parser.add_argument(
        "--target_selection",
        action="store_true",
        default=False,
        help="Enable the fitting of the z-vertex to get the liquid and solid targets",
    )
    parser.add_argument(
        "--solid_target",
        default="C",
        help="Name of solid target",
        type=str,
    )
    parser.add_argument(
        "--log_file",
        default=None,
        help="Name of the log .txt file to save",
        type=str,
    )
    parser.add_argument(
        "--run_info_file",
        default="/home/rmilton/work_dir/rge_datasets/RGE_Runs_charge_luminosities.csv",
        help="Name of the .csv file containing the meta info for the runs",
        type=str,
    )
    parser.add_argument(
        "--run_number",
        default=20150,
        help="Number of run being analyzed",
        type=int,
    )

    flags = parser.parse_args()

    return flags


def main():
    flags = parse_arguments()

    if flags.log_file is not None:
        open(flags.log_file, "w").close()  # Clear log file at start
    # Open data
    parameters = LoadYaml(flags.config, flags.config_directory)
    if flags.input_file_array is not None:
        input_data = flags.input_file_array
    else:
        input_data = flags.input_file
    events_array = open_data(
        data_paths=input_data,
        branches_to_open=parameters["BRANCHES_TO_SAVE"],
        data_tree_name="reconstructed_electrons",
        open_MC=flags.save_MC,
        MC_branches_to_open=(
            parameters["MC_BRANCHES_TO_SAVE"] if flags.save_MC else None
        ),
        MC_tree_name="MC_electrons",
        nmax=flags.nmax,
        log_file=flags.log_file,
    )
    num_EB_electrons = len(events_array)
    # Apply DIS cuts and other basic cuts
    events_array = apply_kinematic_cuts(
        events_array,
        parameters["ELECTRON_KINEMATIC_CUTS"],
        log_file=flags.log_file,
        number_of_initial_electrons=num_EB_electrons,
    )
    # Apply fiducial cuts
    if flags.save_plots:
        os.makedirs(flags.plots_directory, exist_ok=True)
        plot_names = parameters.get("PLOT_TITLES", {})
        if flags.data_name in plot_names:
            plot_title = (
                f"RGE LD2 + {flags.solid_target} : {plot_names[flags.data_name]}"
            )
    else:
        plot_title = None
    events_array = apply_fiducial_cuts(
        events=events_array,
        fiducial_cuts=parameters["ELECTRON_FIDUCIAL_CUTS"],
        save_plots=flags.save_plots,
        plots_directory=flags.plots_directory,
        plot_title=plot_title,
        log_file=flags.log_file,
        number_of_initial_electrons=num_EB_electrons,
    )

    # Apply partial sampling fraction cuts
    events_array = apply_partial_sampling_fraction_cut(
        events=events_array,
        is_simulation=flags.simulation,
        save_plots=flags.save_plots,
        plots_directory=flags.plots_directory,
        plot_title=plot_title,
        log_file=flags.log_file,
        number_of_initial_electrons=num_EB_electrons,
    )
    # Apply SF cuts
    events_array = apply_sampling_fraction_cut(
        events=events_array,
        save_plots=flags.save_plots,
        plots_directory=flags.plots_directory,
        plot_title=plot_title,
        log_file=flags.log_file,
        number_of_initial_electrons=num_EB_electrons,
    )

    if flags.target_selection:
        events_array = apply_target_selection(
            events=events_array,
            solid_target_name=flags.solid_target,
            save_plots=flags.save_plots,
            plots_directory=flags.plots_directory,
            plot_title=plot_title,
            log_file=flags.log_file,
            number_of_initial_electrons=num_EB_electrons,
        )
    # Calculating integrated luminosity within file
    if not flags.simulation:
        run_info_df = CSV_to_df(flags.run_info_file)
        selected_run_info = run_info_df[run_info_df["Run_Number"] == flags.run_number]
        luminosity = selected_run_info["Integrated_Luminosity"].iloc[0]
        total_num_events = selected_run_info["Num_Events"].iloc[0]
        events_array["total_luminosity"] = luminosity
        events_array["total_num_events"] = total_num_events
        fraction_of_events = np.sum(events_array["pass_reco"]) / total_num_events
        events_array["luminosity_after_cuts"] = fraction_of_events * luminosity
    # Save the cut electrons. Should have the option to cut on targets or not
    save_output(
        events_array,
        flags.output_directory,
        flags.output_file,
        parameters["ELECTRON_SELECTION_BRANCHES_TO_SAVE"],
        flags.save_MC,
        parameters["MC_BRANCHES_TO_SAVE"] if flags.save_MC else None,
    )


if __name__ == "__main__":
    main()
