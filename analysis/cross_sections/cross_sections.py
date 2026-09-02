# This script calculates absolute cross sections given the electron candidates from electron_selection.py
# This script has the option to incorporate unfolding into the cross sections
# If unfolding is enabled, a model will be loaded and the weights will be calculated
# The unfolding model needs to be trained already. See unfolding/RGE_unfolding.py to train an unfolding model.

import numpy as np
import argparse
import glob
import os
import sys
import pandas as pd

REPO_TOP_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO_TOP_DIR)
ANALYSIS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ANALYSIS_DIR)
from utils import LoadYaml, open_data
from analysis_dataloader import AnalysisDataloader
import analysis_options
from radiative_corrections import OpenCorrections
from analysis_helpers import calculate_cross_sections, plot_cross_sections


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input_file",
        default="/home/rmilton/work_dir/rge_datasets/phys_val/020150/candidate_electrons/candidate_electrons_020150_final.root",
        help="ROOT file containing candidate electrons after electron selection",
        type=str,
    )
    parser.add_argument(
        "--input_file_array",
        nargs="+",  # one or more values
        default=None,
        help="Multiple ROOT files containing candidate electrons after electron selection. This will be chosen over input_file if given",
        type=str,
    )
    parser.add_argument(
        "--input_directories",
        nargs="+",  # one or more values
        default=None,
        help="Directories containing candidate electrons",
        type=str,
    )
    parser.add_argument(
        "--nmax",
        default=None,
        help="Max number of events to load",
        type=int,
    )
    parser.add_argument(
        "--solid_output_path",
        default="/home/rmilton/work_dir/rge_datasets/phys_val/020150/carbon_cross_sections.csv",
        help="Path of the .csv file to write the solid-target cross sections to",
        type=str,
    )
    parser.add_argument(
        "--deuterium_output_path",
        default="/home/rmilton/work_dir/rge_datasets/phys_val/020150/LD2_cross_sections.csv",
        help="Path of the .csv file to write the LD2 cross sections to",
        type=str,
    )
    parser.add_argument(
        "--config",
        default="analysis_config.yaml",
        help="Basic config file containing general options",
        type=str,
    )
    parser.add_argument(
        "--config_directory",
        default="/home/rmilton/work_dir/RGE_electron_analysis_git/configs/",
        help="Directory containing the config files",
        type=str,
    )
    parser.add_argument(
        "--plots_directory",
        default="/home/rmilton/work_dir/rge_datasets/phys_val/020150/carbon_plots_final/",
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
        "--use_unfolding",
        action="store_true",
        default=False,
        help="Load unfolding weights from trained model",
    )
    parser.add_argument(
        "--simulation_input_file",
        default=None,
        type=str,
        help="Path to simulation file with candidate electrons. Only used if use_unfolding flag is used",
    )
    parser.add_argument(
        "--simulation_input_file_array",
        nargs="+",  # one or more values
        default=None,
        help="Multiple simulation ROOT files containing candidate electrons after electron selection. This will be chosen over simulation_input_file if given",
        type=str,
    )
    parser.add_argument(
        "--MC_nmax",
        default=None,
        type=int,
        help="Max number of simulation events to load",
    )
    parser.add_argument(
        "--use_radiative_corrections",
        action="store_true",
        default=False,
        help="Load and apply radiative corrections to cross sections",
    )
    parser.add_argument(
        "--solid_radiative_corrections_path",
        default="/work/clas12/rmilton/externals/OUT/clasC12.out",
        type=str,
        help="Path to radiative corrections file for solid target",
    )
    parser.add_argument(
        "--deuterium_radiative_corrections_path",
        default="/work/clas12/rmilton/externals/OUT/clasd2.out",
        type=str,
        help="Path to radiative corrections file for deuterium",
    )
    parser.add_argument(
        "--solid_target", default="C", type=str, help="Name of solid target"
    )
    parser.add_argument(
        "--num_processes", default=16, type=int, help="Number of processes"
    )
    parser.add_argument("--log_file", default=None, type=str, help="Name of log file")
    parser.add_argument(
        "--efficiency_file", default=None, type=str, help=".csv file with efficiencies"
    )

    flags = parser.parse_args()

    return flags


def get_input_files(flags):
    """Resolves the candidate-electron input files, in precedence order:
    --input_directories (globbed) > --input_file_array > --input_file."""
    if flags.input_directories is not None:
        input_files = []
        for directory in flags.input_directories:
            input_files.extend(glob.glob(f"{directory}/candidate_electrons/*.root"))
        return sorted(input_files)
    if flags.input_file_array is not None:
        return flags.input_file_array
    return [flags.input_file]


def main():
    flags = parse_arguments()
    # Reading config file
    parameters = LoadYaml(flags.config, flags.config_directory)

    input_files = get_input_files(flags)
    njobs = max(1, min(flags.num_processes, len(input_files)))

    data_array = open_data(
        data_paths=input_files,
        branches_to_open=parameters["BRANCHES_TO_LOAD"],
        data_tree_name="reconstructed_electrons",
        nmax=flags.nmax,
        get_meta_info=True,
        num_processes=njobs,
        log_file=flags.log_file,
    )

    # No train/test split here and histogramming is order-independent, so
    # shuffling would only cost a full permutation copy of the combined array.
    data_dataloader = AnalysisDataloader(
        reconstructed=data_array["reconstructed"],
        MC=None,
        shuffle=False,
    )

    if flags.use_unfolding:
        raise NotImplementedError(
            "--use_unfolding does not work. This branch never calls "
            "unfolding_procedure() (analysis/unfolding/RGE_unfolding.py shows how "
            "it should be called), and it still passes the old open_MC=/"
            "MC_branches_to_open=/MC_tree_name= arguments, which utils.open_data "
            "no longer accepts."
        )
        if flags.input_file_array is not None:
            input_simulation = flags.simulation_input_file_array
        else:
            input_simulation = flags.simulation_input_file

        simulation_array = open_data(
            data_paths=input_simulation,
            branches_to_open=parameters["BRANCHES_TO_LOAD"],
            open_MC=True,
            MC_branches_to_open=parameters["MC_BRANCHES_TO_LOAD"],
            MC_tree_name="MC_electrons",
            nmax=flags.MC_nmax,
            shuffle=True,
        )

        simulation_dataloader = AnalysisDataloader(
            reconstructed=simulation_array["reconstructed"],
            MC=simulation_array["MC"],
            shuffle=True,
        )

        step2_weights = simulation_dataloader["step2_weights"]

    # Load radiative corrections from files
    if flags.use_radiative_corrections:

        radiative_corrections_df_solid = OpenCorrections(
            flags.solid_radiative_corrections_path
        )
        radiative_corrections_df_deuterium = OpenCorrections(
            flags.deuterium_radiative_corrections_path
        )
        radiative_corrections_dictionary = {
            flags.solid_target: radiative_corrections_df_solid,
            "LD2": radiative_corrections_df_deuterium,
        }

    # Calculate cross sections
    # Have options to include unfolding and radiative corrections
    # In the output file, have unfolded and non-unfolded cross sections
    output_dataframes = {flags.solid_target: pd.DataFrame({}), "LD2": pd.DataFrame({})}

    # The luminosity is per-run, not per-target, so this is computed once for
    # both targets. total_luminosity is a per-run constant broadcast to every
    # event, hence the [0] on each run's slice.
    run_numbers = np.asarray(data_array["meta_info"]["run_number"])
    luminosities = np.asarray(data_array["meta_info"]["total_luminosity"])

    unique_runs = np.unique(run_numbers)
    print("Unique run numbers: ", unique_runs)

    total_integrated_luminosity = 0
    for run in unique_runs:
        run_mask = run_numbers == run
        total_integrated_luminosity += luminosities[run_mask][0]

    print(f"Total integrated luminosity: {total_integrated_luminosity}")

    fraction_pass_reco = np.sum(data_dataloader.pass_reco) / len(
        data_dataloader.pass_reco
    )
    print("Fraction pass reco: ", fraction_pass_reco)

    os.makedirs(flags.plots_directory, exist_ok=True)

    for target_name in [flags.solid_target, "LD2"]:
        x_bin_edges = analysis_options.x_bins_by_target[target_name]
        Q2_bin_edges = analysis_options.Q2_bins_by_target[target_name]
        x_bin_centers = (x_bin_edges[1:] + x_bin_edges[:-1]) / 2
        Q2_bin_centers = (Q2_bin_edges[1:] + Q2_bin_edges[:-1]) / 2

        repeated_x_bin_centers = np.repeat(x_bin_centers, len(Q2_bin_centers))
        repeated_Q2_bin_centers = np.tile(Q2_bin_centers, len(x_bin_centers))
        output_dataframes[target_name]["x_bin_center"] = repeated_x_bin_centers
        output_dataframes[target_name]["Q2_bin_center"] = repeated_Q2_bin_centers

        # No unfolding, no radiative corrections
        (
            absolute_cross_section_norad_nounfolding,
            absolute_cross_section_norad_nounfolding_errors,
        ) = calculate_cross_sections(
            dataloader=data_dataloader,
            target_name=target_name,
            x_binning=x_bin_edges,
            Q2_binning=Q2_bin_edges,
            apply_radiative_corrections=False,
            integrated_luminosity=total_integrated_luminosity,
            efficiency_file=flags.efficiency_file,
        )
        output_dataframes[target_name][
            "cross_section_norad_nounfolding"
        ] = absolute_cross_section_norad_nounfolding
        output_dataframes[target_name][
            "cross_section_norad_nounfolding_errors"
        ] = absolute_cross_section_norad_nounfolding_errors
        save_path = None

        if flags.save_plots:
            save_path = f"{flags.plots_directory}/{target_name}_cross_section_norad_nounfolding.png"
        plot_cross_sections(
            output_dataframes[target_name],
            x_binning=x_bin_edges,
            Q2_binning=np.arange(1, 12, 1),
            cross_section_name="cross_section_norad_nounfolding",
            plot_title=f"{target_name}, No rad. corrections, no unfolding",
            save_path=save_path,
        )

        if flags.use_radiative_corrections:
            # No unfolding, with radiative corrections
            (
                absolute_cross_section_withrad_nounfolding,
                absolute_cross_section_withrad_nounfolding_errors,
            ) = calculate_cross_sections(
                dataloader=data_dataloader,
                target_name=target_name,
                x_binning=x_bin_edges,
                Q2_binning=Q2_bin_edges,
                apply_radiative_corrections=True,
                integrated_luminosity=total_integrated_luminosity,
                efficiency_file=flags.efficiency_file,
                radiative_corrections_df=radiative_corrections_dictionary[target_name],
            )
            output_dataframes[target_name][
                "cross_section_withrad_nounfolding"
            ] = absolute_cross_section_withrad_nounfolding
            output_dataframes[target_name][
                "cross_section_withrad_nounfolding_errors"
            ] = absolute_cross_section_withrad_nounfolding_errors

            if flags.save_plots:
                save_path = f"{flags.plots_directory}/{target_name}_cross_section_withrad_nounfolding.png"
            plot_cross_sections(
                output_dataframes[target_name],
                x_bin_edges,
                Q2_binning=np.arange(1, 12, 1),
                cross_section_name="cross_section_withrad_nounfolding",
                plot_title=f"{target_name}, With rad. corrections, no unfolding",
                save_path=save_path,
            )
        if flags.use_unfolding:
            # No unfolding, with radiative corrections
            (
                absolute_cross_section_withrad_withunfolding,
                absolute_cross_section_withrad_withunfolding_errors,
            ) = calculate_cross_sections(
                dataloader=data_dataloader,
                target_name=target_name,
                x_binning=x_bin_edges,
                Q2_binning=Q2_bin_edges,
                apply_radiative_corrections=True,
                integrated_luminosity=total_integrated_luminosity,
                efficiency_file=flags.efficiency_file,
                radiative_corrections_df=radiative_corrections_dictionary[target_name],
                use_truth=True,
                weights=step2_weights,
            )
            output_dataframes[target_name][
                "cross_section_withrad_withunfolding"
            ] = absolute_cross_section_withrad_withunfolding
            output_dataframes[target_name][
                "cross_section_withrad_withunfolding_errors"
            ] = absolute_cross_section_withrad_withunfolding_errors

            if flags.save_plots:
                save_path = f"{flags.plots_directory}/{target_name}_cross_section_withrad_withunfolding.png"
            plot_cross_sections(
                output_dataframes[target_name],
                x_bin_edges,
                Q2_binning=np.arange(1, 12, 1),
                cross_section_name="cross_section_withrad_withunfolding",
                plot_title=f"{target_name}, With rad. corrections, With unfolding",
                save_path=save_path,
            )

    output_dataframes[flags.solid_target].query(
        "cross_section_norad_nounfolding > 0"
    ).to_csv(
        flags.solid_output_path,
        index=False,
    )
    output_dataframes["LD2"].query("cross_section_norad_nounfolding > 0").to_csv(
        flags.deuterium_output_path,
        index=False,
    )


if __name__ == "__main__":
    main()
