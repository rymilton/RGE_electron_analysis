import argparse
import uproot
import awkward as ak
import numpy as np
import os
import concurrent.futures as cf
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
        "--save_gen",
        action="store_true",
        default=False,
        help="Load generator-level (Monte Carlo truth) information from file and save it in output electrons",
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
    parser.add_argument(
        "--develop_cuts",
        default=False,
        help="Set to true if you want to remake the cuts",
        action="store_true",
    )
    parser.add_argument(
        "--cut_directory",
        default="./cuts/C_D2/",
        help="Name of the directory containing the cut json files",
        type=str,
    )
    parser.add_argument(
        "--num_processes",
        default=4,
        help="Number of processes to use when applying cuts to each input file in parallel",
        type=int,
    )

    flags = parser.parse_args()

    return flags


def get_plot_title(flags, parameters):
    if not flags.save_plots:
        return None
    plot_names = parameters.get("PLOT_TITLES", {})
    if flags.data_name in plot_names:
        return f"RGE LD2 + {flags.solid_target} : {plot_names[flags.data_name]}"
    else:
        return f"RGE LD2 + {flags.solid_target} : {flags.run_number} Pass 1"


def get_input_files(flags):
    if flags.input_file_array is not None:
        return flags.input_file_array
    return [flags.input_file]


def run_cut_pipeline(
    events_array,
    flags,
    parameters,
    plot_title,
    develop_cuts,
    save_plots,
    number_of_initial_electrons=None,
):
    """Runs the full kinematic/fiducial/partial-SF/SF/status/target cut
    pipeline on whatever array it's given. develop_cuts toggles whether the
    fit-based cut functions (partial sampling, sampling fraction, target)
    re-fit from this data or load previously-cached cut parameters.
    save_plots is explicit (rather than read from flags) so callers can force
    plotting off independent of the top-level --save_plots flag."""
    if number_of_initial_electrons is None:
        number_of_initial_electrons = len(events_array)

    events_array = apply_kinematic_cuts(
        events_array,
        parameters["ELECTRON_KINEMATIC_CUTS"],
        save_plots=save_plots,
        plots_directory=flags.plots_directory,
        plot_title=plot_title,
        log_file=flags.log_file,
        number_of_initial_electrons=number_of_initial_electrons,
    )
    events_array = apply_fiducial_cuts(
        events=events_array,
        fiducial_cuts=parameters["ELECTRON_FIDUCIAL_CUTS"],
        save_plots=save_plots,
        plots_directory=flags.plots_directory,
        plot_title=plot_title,
        log_file=flags.log_file,
        number_of_initial_electrons=number_of_initial_electrons,
    )
    events_array = apply_partial_sampling_fraction_cut(
        events=events_array,
        develop_cuts=develop_cuts,
        cut_params_path=os.path.join(flags.cut_directory, "partial_sampling.json"),
        is_simulation=flags.simulation,
        save_plots=save_plots,
        plots_directory=flags.plots_directory,
        plot_title=plot_title,
        log_file=flags.log_file,
        number_of_initial_electrons=number_of_initial_electrons,
    )
    events_array = apply_sampling_fraction_cut(
        events=events_array,
        develop_cuts=develop_cuts,
        cut_params_path=os.path.join(flags.cut_directory, "sampling_fraction.json"),
        save_plots=save_plots,
        plots_directory=flags.plots_directory,
        plot_title=plot_title,
        log_file=flags.log_file,
        number_of_initial_electrons=number_of_initial_electrons,
    )
    events_array = apply_status_cut(
        events_array,
        log_file=flags.log_file,
        number_of_initial_electrons=number_of_initial_electrons,
    )
    if flags.target_selection:
        events_array = apply_target_selection(
            events=events_array,
            solid_target_name=flags.solid_target,
            develop_cuts=develop_cuts,
            cut_params_path=os.path.join(flags.cut_directory, "target.json"),
            save_plots=save_plots,
            plots_directory=flags.plots_directory,
            plot_title=plot_title,
            log_file=flags.log_file,
            number_of_initial_electrons=number_of_initial_electrons,
        )

    return events_array, number_of_initial_electrons


def _pad_failed_trigger_events(failed_arr, n, target_selection):
    """Explicit defaults for the fields run_cut_pipeline adds, since these
    events skip the pipeline entirely (see run_cut_pipeline_respecting_trigger)."""
    false_mask = np.zeros(n, dtype=np.bool_)
    for field in (
        "pass_reco",
        "pass_kinematic",
        "pass_fiducial",
        "pass_partial_SF",
        "pass_SF",
    ):
        failed_arr = ak.with_field(failed_arr, false_mask, field)

    failed_arr["reconstructed"] = ak.with_field(
        failed_arr["reconstructed"], np.full(n, -9999.0), "total_ecal_energy"
    )
    if target_selection:
        failed_arr["reconstructed"] = ak.with_field(
            failed_arr["reconstructed"], ["None"] * n, "target"
        )
    return failed_arr


def run_cut_pipeline_respecting_trigger(
    events_array, flags, parameters, plot_title, develop_cuts, save_plots
):
    """Only feeds events with has_trigger_electron == True into the cut
    pipeline. If has_trigger_electron isn't in the file at all (older
    files), falls back to running cuts on everything, unchanged from
    before. Events that fail has_trigger_electron are kept in the output
    as-is, tagged pass_trigger=False and backfilled with sensible defaults
    for every field the cut pipeline would otherwise have added."""
    if "has_trigger_electron" not in events_array["reconstructed"].fields:
        return run_cut_pipeline(
            events_array, flags, parameters, plot_title, develop_cuts, save_plots
        )

    number_of_initial_electrons = len(events_array)
    events_array["pass_trigger"] = events_array["reconstructed"]["has_trigger_electron"]

    # Tag original row order so it can be restored after the split/merge
    events_array = ak.with_field(
        events_array, np.arange(len(events_array)), "_original_index"
    )

    trigger_mask = ak.values_astype(events_array["pass_trigger"], np.bool_)
    passed = events_array[trigger_mask]
    failed = events_array[~trigger_mask]

    passed, _ = run_cut_pipeline(
        passed,
        flags,
        parameters,
        plot_title,
        develop_cuts,
        save_plots,
        number_of_initial_electrons=number_of_initial_electrons,
    )

    failed = _pad_failed_trigger_events(failed, len(failed), flags.target_selection)

    combined = ak.concatenate([passed, failed])
    combined = combined[ak.argsort(combined["_original_index"])]
    combined = combined[[f for f in combined.fields if f != "_original_index"]]

    return combined, number_of_initial_electrons


def add_luminosity_info(events_array, flags):
    if flags.simulation:
        return events_array
    run_info_df = CSV_to_df(flags.run_info_file)
    selected_run_info = run_info_df[run_info_df["Run_Number"] == flags.run_number]
    luminosity = selected_run_info["Integrated_Luminosity"].iloc[0]
    total_num_events = selected_run_info["Num_Events"].iloc[0]
    events_array["total_luminosity"] = luminosity
    events_array["total_num_events"] = total_num_events
    fraction_of_events = np.sum(events_array["pass_reco"]) / total_num_events
    events_array["luminosity_after_cuts"] = fraction_of_events * luminosity
    events_array["run_number"] = flags.run_number
    return events_array


def derive_output_file_name(path):
    base_name = os.path.splitext(os.path.basename(path))[0]
    if base_name.startswith("electrons_eventbuilder_"):
        base_name = base_name[len("electrons_eventbuilder_") :]
    return f"candidate_electrons_{base_name}.root"


def apply_and_save_one_file(path, flags, parameters, plot_title):
    """Pass 2: opens one file on its own, applies the already-known cuts
    (develop_cuts=False -- the cut JSONs are guaranteed to exist by now,
    whether freshly fit or pre-existing) without making any plots (those
    were already made from the combined array in pass 1), and saves."""
    events_array = open_data(
        data_paths=path,
        branches_to_open=parameters["BRANCHES_TO_SAVE"],
        data_tree_name="reconstructed_electrons",
        open_gen=flags.save_gen,
        gen_branches_to_open=(
            parameters["GEN_BRANCHES_TO_SAVE"] if flags.save_gen else None
        ),
        gen_tree_name="gen_electrons",
        nmax=flags.nmax,
        log_file=flags.log_file,
    )
    events_array, _ = run_cut_pipeline_respecting_trigger(
        events_array,
        flags,
        parameters,
        plot_title,
        develop_cuts=False,
        save_plots=False,
    )
    events_array = add_luminosity_info(events_array, flags)

    if flags.input_file_array is not None:
        output_file = derive_output_file_name(path)
    else:
        output_file = flags.output_file

    save_output(
        events_array,
        flags.output_directory,
        output_file,
        parameters["ELECTRON_SELECTION_BRANCHES_TO_SAVE"],
        flags.save_gen,
        parameters["GEN_BRANCHES_TO_SAVE"] if flags.save_gen else None,
    )
    return output_file


def main():
    flags = parse_arguments()

    if flags.log_file is not None:
        open(flags.log_file, "w").close()  # Clear log file at start
    parameters = LoadYaml(flags.config, flags.config_directory)
    os.makedirs(flags.cut_directory, exist_ok=True)
    if flags.save_plots:
        os.makedirs(flags.plots_directory, exist_ok=True)

    input_files = get_input_files(flags)
    plot_title = get_plot_title(flags, parameters)

    # Pass 1: combine all input files, run the cut pipeline once so the
    # diagnostic plots reflect the whole run's combined statistics and (if
    # --develop_cuts) the cut parameters are fit from the full combined
    # dataset. Nothing from this pass is saved -- it exists only to drive
    # plots and cut development.
    print(f"Combining {len(input_files)} file(s) for diagnostics...")
    events_array = open_data(
        data_paths=input_files,
        branches_to_open=parameters["BRANCHES_TO_SAVE"],
        data_tree_name="reconstructed_electrons",
        open_gen=flags.save_gen,
        gen_branches_to_open=(
            parameters["GEN_BRANCHES_TO_SAVE"] if flags.save_gen else None
        ),
        gen_tree_name="gen_electrons",
        nmax=flags.nmax,
        log_file=flags.log_file,
    )
    events_array, number_of_initial_electrons = run_cut_pipeline_respecting_trigger(
        events_array,
        flags,
        parameters,
        plot_title,
        develop_cuts=flags.develop_cuts,
        save_plots=flags.save_plots,
    )
    if flags.develop_cuts:
        print(
            f"Cuts developed and saved to {flags.cut_directory} "
            f"({ak.sum(events_array['pass_reco'])}/{number_of_initial_electrons} pass after all cuts)"
        )
    del events_array  # combined array was only used for diagnostics/fitting above

    # Pass 2: apply the now-known cuts to each input file independently, in
    # parallel, saving one output file per input file.
    os.makedirs(flags.output_directory, exist_ok=True)
    njobs = max(1, min(flags.num_processes, len(input_files)))
    if njobs == 1:
        for path in input_files:
            try:
                output_file = apply_and_save_one_file(path, flags, parameters, plot_title)
                print(f"Saved {output_file}")
            except Exception as e:
                print(f"FAILED: {path}: {e}")
    else:
        with cf.ProcessPoolExecutor(max_workers=njobs) as executor:
            futures = {
                executor.submit(
                    apply_and_save_one_file, path, flags, parameters, plot_title
                ): path
                for path in input_files
            }
            for future in cf.as_completed(futures):
                path = futures[future]
                try:
                    output_file = future.result()
                    print(f"Saved {output_file}")
                except Exception as e:
                    print(f"FAILED: {path}: {e}")


if __name__ == "__main__":
    main()
