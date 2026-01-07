import ROOT
import argparse
import numpy as np
import uproot as ur
import awkward as ak
import os
import matplotlib.pyplot as plt
import mplhep as hep
hep.style.use("CMS")
hep.style.use(hep.style.CMS)
import matplotlib.colors as mcolors
import sys
import yaml
from cycler import cycler
plt.rcParams["axes.prop_cycle"] = cycler('color', ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'])

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis_helpers import unfolding_procedure, plot_unfolded
from analysis_dataloader import AnalysisDataloader
sys.path.append("../..")
from utils import open_data

def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--roounfold_path",
        default="/home/rmilton/work_dir/unbinned_unfolding/build/RooUnfold/",
        help="Path to the RooUnfold build directory",
        type=str,
    )
    parser.add_argument(
        "--load_omnifold_model",
        default=False,
        help="Whether to load a trained omnifold model",
        action="store_true"
    )
    parser.add_argument(
        "--model_path",
        default=None,
        help="Path to the trained omnifold model",
        type=str
    )
    parser.add_argument(
        "--num_iterations",
        default=4,
        help="Number of omnifold iterations to perform",
        type=int
    )
    parser.add_argument(
        "--num_events",
        default=5000000,
        help="Number of events to use",
        type=int
    )
    parser.add_argument(
        "--MC_name",
        default="GiBUU",
        help="Name of the dataset to use as the simulation/Monte Carlo sample",
        type=str
    )
    parser.add_argument(
        "--pseudodata_name",
        default="clasdis",
        help="Name of the dataset to use as the pseudodata sample",
        type=str
    )
    parser.add_argument(
        "--config",
        default="/home/rmilton/work_dir/RGE_electron_analysis_git/analysis/unfolding_config.yaml",
        help="Path to the YAML unfolding configuration file",
        type=str
    )
    parser.add_argument(
        "--train_test_split",
        default=True,
        help="Whether to split the data into training and testing sets",
        action="store_true"
    )
    parser.add_argument(
        "--test_fraction",
        default=0.2,
        help="Fraction of data to use for testing",
        type=float
    )
    parser.add_argument(
        "--plot_directory",
        default="/home/rmilton/work_dir/RGE_electron_analysis_git/analysis/unfolding/plots/",
        help="Path to the directory where plots will be saved",
        type=str
    )

    flags = parser.parse_args()

    return flags

def main():
    flags = parse_arguments()
    parameters = yaml.safe_load(open(flags.config))

    # Setting up omnifold
    sys.path.append(flags.roounfold_path)
    from omnifold import OmniFold_helper_functions

    load_trained_omnifold_model = flags.load_omnifold_model
    total_number_of_events = flags.num_events

    plot_title = f"{flags.MC_name} vs. {flags.pseudodata_name}: C/LD2"

    MC_BRANCHES_TO_LOAD = [
        "MC_px",
        "MC_py",
        "MC_pz",
        "MC_vx",
        "MC_vy",
        "MC_vz",
        "MC_theta_degrees",
        "MC_phi_degrees",
        "MC_Q2",
        "MC_nu",
        "MC_x",
        "MC_y",
        "MC_W",
    ]

    BRANCHES_TO_LOAD = [
        "p_x",
        "p_y",
        "p_z",
        "p",
        "theta_degrees",
        "phi_degrees",
        "v_x",
        "v_y",
        "v_z",
        "Q2",
        "nu",
        "x",
        "y",
        "W",
        "pass_reco"
    ]

    valid_montecarlos = ["GiBUU", "clasdis"]
    if flags.MC_name not in valid_montecarlos:
        raise ValueError(f"Invalid MC_name: {flags.MC_name}. Must be one of {valid_montecarlos}")
    if flags.pseudodata_name not in valid_montecarlos:
        raise ValueError(f"Invalid pseudodata_name: {flags.pseudodata_name}. Must be one of {valid_montecarlos}")
    print(f"Using MC: {flags.MC_name}, Pseudodata: {flags.pseudodata_name}")
    files_dict = {
        "GiBUU_solid": parameters["GIBUU_SOLID_FILE"],
        "GiBUU_liquid": parameters["GIBUU_LIQUID_FILE"],
        "clasdis_solid": parameters["CLASDIS_SOLID_FILE"],
        "clasdis_liquid": parameters["CLASDIS_LIQUID_FILE"]
    }
    simulation_files, pseudodata_files = {}, {}
    for name in files_dict:
        if flags.MC_name in name:
            if "solid" in name:
                simulation_files["solid"] = files_dict[name]
            else:
                simulation_files["liquid"] = files_dict[name]
        if flags.pseudodata_name in name:
            if "solid" in name:
                pseudodata_files["solid"] = files_dict[name]
            else:
                pseudodata_files["liquid"] = files_dict[name]

    simulation_event_dictionary = {"reconstructed_liquid": ak.Array([]), "MC_liquid": ak.Array([]), "reconstructed_solid": ak.Array([]), "MC_solid": ak.Array([])}
    pseudodata_event_dictionary = {"reconstructed_liquid": ak.Array([]), "MC_liquid": ak.Array([]), "reconstructed_solid": ak.Array([]), "MC_solid": ak.Array([])}
    
    # Opening simulation file
    for name in simulation_files:
        simulation_events = open_data(
            simulation_files[name],
            BRANCHES_TO_LOAD,
            data_tree_name="reconstructed_electrons",
            open_MC=True,
            MC_branches_to_open=MC_BRANCHES_TO_LOAD,
            MC_tree_name="MC_electrons",
            nmax=flags.num_events,
            output_format="dictionary"
        )
        if "solid" in name:
            simulation_event_dictionary["reconstructed_solid"] = simulation_events["reconstructed"]
            simulation_event_dictionary["MC_solid"] = simulation_events["MC"]
        else:
            simulation_event_dictionary["reconstructed_liquid"] = simulation_events["reconstructed"]
            simulation_event_dictionary["MC_liquid"] = simulation_events["MC"]
    # Opening pseudodata file
    for name in pseudodata_files:
        pseudodata_events = open_data(
            pseudodata_files[name],
            BRANCHES_TO_LOAD,
            data_tree_name="reconstructed_electrons",
            open_MC=True,
            MC_branches_to_open=MC_BRANCHES_TO_LOAD,
            MC_tree_name="MC_electrons",
            nmax=flags.num_events,
            output_format="dictionary"
        )
        if "solid" in name:
            pseudodata_event_dictionary["reconstructed_solid"] = pseudodata_events["reconstructed"]
            pseudodata_event_dictionary["MC_solid"] = pseudodata_events["MC"]
        else:
            pseudodata_event_dictionary["reconstructed_liquid"] = pseudodata_events["reconstructed"]
            pseudodata_event_dictionary["MC_liquid"] = pseudodata_events["MC"]

    # Setting up simulation dataloader
    simulation_event_dictionary["reconstructed_solid"]["weight"] = 1
    simulation_event_dictionary["reconstructed_liquid"]["weight"] = 1
    
    simulation_reconstructed = ak.concatenate([simulation_event_dictionary["reconstructed_solid"], simulation_event_dictionary["reconstructed_liquid"]])
    simulation_MC = ak.concatenate([simulation_event_dictionary["MC_solid"], simulation_event_dictionary["MC_liquid"]])

    simulation_dataloader = AnalysisDataloader(
        reconstructed = simulation_reconstructed,
        MC = simulation_MC,
        shuffle = True,
        max_num_events = total_number_of_events,
        train_test_split = flags.train_test_split,
        test_fraction = flags.test_fraction,
        name = flags.MC_name,
    )

    # Setting up pseudodata dataloader
    pseudodata_event_dictionary["reconstructed_solid"]["weight"] = 1
    pseudodata_event_dictionary["reconstructed_liquid"]["weight"] = 1
    pseudodata_reconstructed = ak.concatenate([pseudodata_event_dictionary["reconstructed_solid"], pseudodata_event_dictionary["reconstructed_liquid"]])
    pseudodata_MC = ak.concatenate([pseudodata_event_dictionary["MC_solid"], pseudodata_event_dictionary["MC_liquid"]])

    pseudodata_dataloader = AnalysisDataloader(
        reconstructed = pseudodata_reconstructed,
        MC = pseudodata_MC,
        shuffle = True,
        max_num_events = total_number_of_events,
        train_test_split = flags.train_test_split,
        test_fraction = flags.test_fraction,
        name = flags.pseudodata_name,
    )

    # Making plots directory if it doesn't exist
    if not os.path.exists(flags.plot_directory):
        os.makedirs(flags.plot_directory)

    # Plots to compare the phase spaces of the two distributios
    fig = plt.figure()
    plt.hist2d(np.array(simulation_dataloader.MC["MC_x"]), np.array(simulation_dataloader.MC["MC_Q2"]), bins=100, range=((0,1), (1,12)), label="simulation", norm=mcolors.LogNorm())
    plt.xlabel("x")
    plt.ylabel("$Q^2~(GeV^2)$")
    plt.colorbar()
    plt.suptitle(f"{simulation_dataloader.data_name} Truth-level phase space")
    # Saving figure to plots_dir
    plt.savefig(os.path.join(flags.plot_directory, f"{simulation_dataloader.data_name}_truth_phasespace.png"))

    fig = plt.figure()
    plt.hist2d(np.array(pseudodata_dataloader.MC["MC_x"]), np.array(pseudodata_dataloader.MC["MC_Q2"]), bins=100, range=((0,1), (1,12)), label="GiBUU", norm=mcolors.LogNorm())
    plt.xlabel("x")
    plt.ylabel("$Q^2~(GeV^2)$")
    plt.colorbar()
    plt.suptitle(f"{pseudodata_dataloader.data_name} Truth-level phase space")
    plt.savefig(os.path.join(flags.plot_directory, f"{pseudodata_dataloader.data_name}_truth_phasespace.png"))

    fig = plt.figure()
    plt.hist2d(np.array(simulation_dataloader.MC["MC_x"][simulation_dataloader.pass_truth]), np.array(simulation_dataloader.MC["MC_Q2"][simulation_dataloader.pass_truth]), bins=100, range=((0,1), (1,12)), label="clasdis", norm=mcolors.LogNorm())
    plt.xlabel("x")
    plt.ylabel("$Q^2~(GeV^2)$")
    plt.colorbar()
    plt.suptitle(f"{simulation_dataloader.data_name} Truth-level phase space")
    plt.savefig(os.path.join(flags.plot_directory, f"{simulation_dataloader.data_name}_truth_phasespace_passtruth.png"))

    fig = plt.figure()
    plt.hist2d(np.array(pseudodata_dataloader.MC["MC_x"][pseudodata_dataloader.pass_truth]), np.array(pseudodata_dataloader.MC["MC_Q2"][pseudodata_dataloader.pass_truth]), bins=100, range=((0,1), (1,12)), label="GiBUU", norm=mcolors.LogNorm())
    plt.xlabel("x")
    plt.ylabel("$Q^2~(GeV^2)$")
    plt.colorbar()
    plt.suptitle(f"{pseudodata_dataloader.data_name} Truth-level phase space")
    plt.savefig(os.path.join(flags.plot_directory, f"{pseudodata_dataloader.data_name}_truth_phasespace_passtruth.png"))

    variables_to_unfold = parameters["UNFOLDING_VARIABLES"]
    print(f"Unfolding with variables: {variables_to_unfold}")

    step1_weights, step2_weights = unfolding_procedure(
        flags,
        simulation_dataloader,
        pseudodata_dataloader,
        variables_to_unfold,
    )

    plt.figure()
    plt.hist(step1_weights, bins=100)
    plt.xlabel("Iteration 1 Step 1 weights")
    plt.ylabel("Counts")
    plt.title(plot_title)
    plt.savefig(os.path.join(flags.plot_directory, "gibuu_clasdis_closure_iteration1_step1weights.png"))

    plt.figure()
    plt.hist(step2_weights, bins=100)
    plt.xlabel(f"Iteration {flags.num_iterations} Step 2 weights")
    plt.ylabel("Counts")
    plt.title(plot_title)
    plt.savefig(os.path.join(flags.plot_directory, f"gibuu_clasdis_closure_iteration{flags.num_iterations}_step2weights.png"))

    # Define binning and labels for each variable
    variable_settings = {
        "p":    {"bins": 50, "range": (0, 10), "xlabel": "$p~(GeV)$"},
        "theta_degrees": {"bins": 50, "range": (0, 60), "xlabel": r"$\theta~(deg)$"},
        "phi_degrees":   {"bins": 50, "range": (-180, 180), "xlabel": r"$\phi~(deg)$"},
        "x":     {"bins": 50, "range": (0, 1), "xlabel": "$x$"},
        "Q2":    {"bins": 50, "range": (0, 10), "xlabel": "$Q^2~(GeV^2)$"},
        "vz":   {"bins": 50, "range": (-12, 5), "xlabel": "$v_z ~(cm)$"},
    }

    for var in variables_to_unfold:
        cfg = variable_settings[var]
        plot_unfolded(
            simulation_dataloader.reconstructed[var][simulation_dataloader.pass_reco_test],
            pseudodata_dataloader.reconstructed[var][pseudodata_dataloader.pass_reco_test],
            step1_weights[simulation_dataloader.pass_reco_test],
            f"{simulation_dataloader.data_name} Reconstructed",
            f"{pseudodata_dataloader.data_name} Reconstructed",
            f"{simulation_dataloader.data_name} with Step 1 weights",
            cfg["bins"], cfg["range"],
            cfg["xlabel"], plot_title,
            f"{flags.plot_directory}/{flags.MC_name}_{flags.pseudodata_name}_closure_iteration1_step1_{var}.png"
        )

        plot_unfolded(
            simulation_dataloader.MC["MC_" + var][simulation_dataloader.pass_truth_test],
            pseudodata_dataloader.MC["MC_" + var][pseudodata_dataloader.pass_truth_test],
            step2_weights[simulation_dataloader.pass_truth_test],
            f"{simulation_dataloader.data_name} Truth",
            f"{pseudodata_dataloader.data_name} Truth",
            f"Unfolded {pseudodata_dataloader.data_name}",
            cfg["bins"], cfg["range"],
            cfg["xlabel"], plot_title,
            f"{flags.plot_directory}/{flags.MC_name}_{flags.pseudodata_name}_closure_iteration{flags.num_iterations}_step2_{var}.png"
        )

if __name__ == "__main__":
    main()