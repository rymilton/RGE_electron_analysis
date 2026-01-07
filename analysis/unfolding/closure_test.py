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
from analysis_helpers import DivideWithErrors, np_to_TVector, TVector_to_np, plot_unfolded
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

    # Setting up simulation awkward array
    simulation_event_dictionary["reconstructed_solid"]["weight"] = 1
    simulation_event_dictionary["reconstructed_liquid"]["weight"] = 1
    simulation_event_dictionary["MC_solid"]["MC_p"] = np.sqrt(simulation_event_dictionary["MC_solid"]["MC_px"]**2+simulation_event_dictionary["MC_solid"]["MC_py"]**2+simulation_event_dictionary["MC_solid"]["MC_pz"]**2)
    simulation_event_dictionary["MC_liquid"]["MC_p"] = np.sqrt(simulation_event_dictionary["MC_liquid"]["MC_px"]**2+simulation_event_dictionary["MC_liquid"]["MC_py"]**2+simulation_event_dictionary["MC_liquid"]["MC_pz"]**2)

    simulation_event_arrays = {}
    simulation_event_arrays["reconstructed"] = ak.concatenate([simulation_event_dictionary["reconstructed_solid"], simulation_event_dictionary["reconstructed_liquid"]])
    simulation_event_arrays["MC"] = ak.concatenate([simulation_event_dictionary["MC_solid"], simulation_event_dictionary["MC_liquid"]])
    simulation_event_arrays["reconstructed"]["vz"] = simulation_event_arrays["reconstructed"]["v_z"]
    simulation_event_arrays["MC"]["pass_truth"] = (simulation_event_arrays["MC"]["MC_p"]>2) & (simulation_event_arrays["MC"]["MC_p"]<8) & (simulation_event_arrays["MC"]["MC_W"]>2) & (simulation_event_arrays["MC"]["MC_y"]<0.8) & (simulation_event_arrays["MC"]["MC_theta_degrees"]>5)
    simulation_event_arrays = ak.Array(simulation_event_arrays)

    # Shuffling to remove any sector ordering that might be present
    permutation = np.random.permutation(len(simulation_event_arrays))
    simulation_event_arrays = simulation_event_arrays[permutation][:total_number_of_events]

    # Splitting into train/test. If we don't split, just copy all events into train and test variables for ease later in code with variable names
    if flags.train_test_split:
        num_simulation_train_events = int((1-flags.test_fraction)*len(simulation_event_arrays))
        simulation_event_arrays_train = simulation_event_arrays[:num_simulation_train_events]
        simulation_event_arrays_test = simulation_event_arrays[num_simulation_train_events:]
        simulation_pass_truth_train = simulation_event_arrays_train["MC"]["pass_truth"]
        simulation_pass_reco_train = simulation_event_arrays_train["reconstructed"]["pass_reco"]
        simulation_pass_truth_test = simulation_event_arrays_test["MC"]["pass_truth"]
        simulation_pass_reco_test = simulation_event_arrays_test["reconstructed"]["pass_reco"]
    else:
        simulation_event_arrays_train = simulation_event_arrays
        simulation_event_arrays_test = simulation_event_arrays
        simulation_pass_truth_train = simulation_event_arrays_train["MC"]["pass_truth"] 
        simulation_pass_reco_train = simulation_event_arrays_train["reconstructed"]["pass_reco"]
        simulation_pass_truth_test = simulation_pass_truth_train
        simulation_pass_reco_test = simulation_pass_reco_train
    
    # Setting up pseudodata awkward array
    pseudodata_event_dictionary["reconstructed_solid"]["weight"] = 1
    pseudodata_event_dictionary["reconstructed_liquid"]["weight"] = 1
    pseudodata_event_dictionary["MC_solid"]["MC_p"] = np.sqrt(pseudodata_event_dictionary["MC_solid"]["MC_px"]**2+pseudodata_event_dictionary["MC_solid"]["MC_py"]**2+pseudodata_event_dictionary["MC_solid"]["MC_pz"]**2)
    pseudodata_event_dictionary["MC_liquid"]["MC_p"] = np.sqrt(pseudodata_event_dictionary["MC_liquid"]["MC_px"]**2+pseudodata_event_dictionary["MC_liquid"]["MC_py"]**2+pseudodata_event_dictionary["MC_liquid"]["MC_pz"]**2)

    pseudodata_event_arrays = {}
    pseudodata_event_arrays["reconstructed"] = ak.concatenate([pseudodata_event_dictionary["reconstructed_solid"], pseudodata_event_dictionary["reconstructed_liquid"]])
    pseudodata_event_arrays["MC"] = ak.concatenate([pseudodata_event_dictionary["MC_solid"], pseudodata_event_dictionary["MC_liquid"]])
    pseudodata_event_arrays["reconstructed"]["vz"] = pseudodata_event_arrays["reconstructed"]["v_z"]
    pseudodata_event_arrays["MC"]["pass_truth"] = (pseudodata_event_arrays["MC"]["MC_p"]>2) & (pseudodata_event_arrays["MC"]["MC_p"]<8) & (pseudodata_event_arrays["MC"]["MC_W"]>2) & (pseudodata_event_arrays["MC"]["MC_y"]<0.8) & (pseudodata_event_arrays["MC"]["MC_theta_degrees"]>5)
    pseudodata_event_arrays = ak.Array(pseudodata_event_arrays)

    # Shuffling to remove any sector ordering that might be present
    permutation = np.random.permutation(len(pseudodata_event_arrays))
    pseudodata_event_arrays = pseudodata_event_arrays[permutation][:total_number_of_events]

    # Splitting into train/test. If we don't split, just copy all events into train and test variables for ease later in code with variable names
    if flags.train_test_split:
        num_simulation_train_events = int((1-flags.test_fraction)*len(pseudodata_event_arrays))
        pseudodata_event_arrays_train = pseudodata_event_arrays[:num_simulation_train_events]
        pseudodata_event_arrays_test = pseudodata_event_arrays[num_simulation_train_events:]
        pseudodata_pass_truth_train = pseudodata_event_arrays_train["MC"]["pass_truth"]
        pseudodata_pass_reco_train = pseudodata_event_arrays_train["reconstructed"]["pass_reco"]
        pseudodata_pass_truth_test = pseudodata_event_arrays_test["MC"]["pass_truth"]
        pseudodata_pass_reco_test = pseudodata_event_arrays_test["reconstructed"]["pass_reco"]
    else:
        pseudodata_event_arrays_train = pseudodata_event_arrays
        pseudodata_event_arrays_test = pseudodata_event_arrays
        pseudodata_pass_truth_train = pseudodata_event_arrays_train["MC"]["pass_truth"]
        pseudodata_pass_reco_train = pseudodata_event_arrays_train["reconstructed"]["pass_reco"]
        pseudodata_pass_truth_test = pseudodata_pass_truth_train
        pseudodata_pass_reco_test = pseudodata_pass_reco_train
    
    # Making plots directory if it doesn't exist
    if not os.path.exists(flags.plot_directory):
        os.makedirs(flags.plot_directory)

    # Plots to compare the phase spaces of the two distributios
    fig = plt.figure()
    plt.hist2d(np.array(simulation_event_arrays["MC"]["MC_x"]), np.array(simulation_event_arrays["MC"]["MC_Q2"]), bins=100, range=((0,1), (1,12)), label="simulation", norm=mcolors.LogNorm())
    plt.xlabel("x")
    plt.ylabel("$Q^2~(GeV^2)$")
    plt.colorbar()
    plt.suptitle(f"{flags.MC_name} Truth-level phase space")
    # Saving figure to plots_dir
    plt.savefig(os.path.join(flags.plot_directory, f"{flags.MC_name}_truth_phasespace.png"))

    fig = plt.figure()
    plt.hist2d(np.array(pseudodata_event_arrays["MC"]["MC_x"]), np.array(pseudodata_event_arrays["MC"]["MC_Q2"]), bins=100, range=((0,1), (1,12)), label="GiBUU", norm=mcolors.LogNorm())
    plt.xlabel("x")
    plt.ylabel("$Q^2~(GeV^2)$")
    plt.colorbar()
    plt.suptitle(f"{flags.MC_name} Truth-level phase space")
    plt.savefig(os.path.join(flags.plot_directory, f"{flags.MC_name}_truth_phasespace.png"))

    fig = plt.figure()
    plt.hist2d(np.array(simulation_event_arrays["MC"]["MC_x"][simulation_pass_truth_train]), np.array(simulation_event_arrays["MC"]["MC_Q2"][simulation_pass_truth_train]), bins=100, range=((0,1), (1,12)), label="clasdis", norm=mcolors.LogNorm())
    plt.xlabel("x")
    plt.ylabel("$Q^2~(GeV^2)$")
    plt.colorbar()
    plt.suptitle(f"{flags.MC_name} Truth-level phase space")
    plt.savefig(os.path.join(flags.plot_directory, f"{flags.MC_name}_truth_phasespace_passtruth.png"))

    fig = plt.figure()
    plt.hist2d(np.array(pseudodata_event_arrays["MC"]["MC_x"][pseudodata_pass_truth_train]), np.array(pseudodata_event_arrays["MC"]["MC_Q2"][pseudodata_pass_truth_train]), bins=100, range=((0,1), (1,12)), label="GiBUU", norm=mcolors.LogNorm())
    plt.xlabel("x")
    plt.ylabel("$Q^2~(GeV^2)$")
    plt.colorbar()
    plt.suptitle(f"{flags.MC_name} Truth-level phase space")
    plt.savefig(os.path.join(flags.plot_directory, f"{flags.MC_name}_truth_phasespace_passtruth.png"))

    variables_to_unfold = parameters["UNFOLDING_VARIABLES"]
    print(f"Unfolding with variables: {variables_to_unfold}")

    
    if not flags.load_omnifold_model:
        print("Setting up training data dictionaries")
        sim_MCreco_dict_train, sim_MCgen_dict_train, data_dict_train = {}, {}, {}
        for variable in variables_to_unfold:
            sim_MCreco_dict_train[variable] = np.array(simulation_event_arrays_train["reconstructed"][variable])
            sim_MCgen_dict_train[variable] = np.array(simulation_event_arrays_train["MC"][f"MC_{variable}"])
            data_dict_train[variable] = np.array(pseudodata_event_arrays_train["reconstructed"][variable])
        df_MCgen_train = ROOT.RDF.FromNumpy(sim_MCgen_dict_train)
        df_MCreco_train = ROOT.RDF.FromNumpy(sim_MCreco_dict_train)
        df_measured_train = ROOT.RDF.FromNumpy(data_dict_train)
        sim_pass_reco_vector_train = np_to_TVector(simulation_pass_reco_train)
        data_pass_reco_vector_train = np_to_TVector(pseudodata_pass_reco_train)

        print("Training omnifold model")
        unbinned_unfolding = ROOT.RooUnfoldOmnifold()
        unbinned_unfolding.SetSaveDirectory("./")
        unbinned_unfolding.SetModelSaveName("clasdis_gibuu_closure")
        unbinned_unfolding.SetMCgenDataFrame(df_MCgen_train)
        unbinned_unfolding.SetMCrecoDataFrame(df_MCreco_train)
        unbinned_unfolding.SetMCPassReco(sim_pass_reco_vector_train)
        unbinned_unfolding.SetMeasuredDataFrame(df_measured_train)
        unbinned_unfolding.SetMeasuredPassReco(data_pass_reco_vector_train)
        unbinned_unfolding.SetNumIterations(4)
        unbinned_results = unbinned_unfolding.UnbinnedOmnifold()

    sim_MCreco_dict_test, sim_MCgen_dict_test, data_dict_test = {}, {}, {}
    for variable in variables_to_unfold:
        sim_MCreco_dict_test[variable] = np.array(simulation_event_arrays_test["reconstructed"][variable])
        sim_MCgen_dict_test[variable] = np.array(simulation_event_arrays_test["MC"]["MC_"+variable])
        data_dict_test[variable] = np.array(pseudodata_event_arrays_test["reconstructed"][variable])
    df_MCgen_test = ROOT.RDF.FromNumpy(sim_MCgen_dict_test)
    df_MCreco_test = ROOT.RDF.FromNumpy(sim_MCreco_dict_test)
    df_measured_test = ROOT.RDF.FromNumpy(data_dict_test)
    sim_pass_reco_vector_test = np_to_TVector(simulation_pass_reco_test)
    data_pass_reco_vector_test = np_to_TVector(pseudodata_pass_reco_test)

    model_name = "clasdis_gibuu_closure" if flags.model_path is None else flags.model_path
    
    unbinned_unfolding = ROOT.RooUnfoldOmnifold()
    unbinned_unfolding.SetTestMCgenDataFrame(df_MCgen_test)
    unbinned_unfolding.SetTestMCrecoDataFrame(df_MCreco_test)
    unbinned_unfolding.SetTestMCPassReco(sim_pass_reco_vector_test)
    unbinned_unfolding.SetLoadModelPath(f"{model_name}_iteration_0.pkl")
    test_unbinned_results = unbinned_unfolding.TestUnbinnedOmnifold()
    step1_weights = TVector_to_np(ROOT.std.get[0](test_unbinned_results))
    plt.figure()
    plt.hist(step1_weights, bins=100)
    plt.xlabel("Step 1 weights before testing")
    plt.savefig(os.path.join(flags.plot_directory, "gibuu_clasdis_closure_iteration1_step1weights.png"))

    unbinned_unfolding.SetTestMCgenDataFrame(df_MCgen_test)
    unbinned_unfolding.SetTestMCrecoDataFrame(df_MCreco_test)
    unbinned_unfolding.SetTestMCPassReco(sim_pass_reco_vector_test)
    unbinned_unfolding.SetLoadModelPath(f"{model_name}_iteration_3.pkl")
    test_unbinned_results = unbinned_unfolding.TestUnbinnedOmnifold()
    step2_weights = TVector_to_np(ROOT.std.get[1](test_unbinned_results))
    plt.figure()
    plt.hist(step2_weights, bins=100)
    plt.xlabel("Step 2 weights after testing")
    plt.savefig(os.path.join(flags.plot_directory, "gibuu_clasdis_closure_iteration4_step2weights.png"))

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
            simulation_event_arrays_test["reconstructed"][var][simulation_pass_reco_test],
            pseudodata_event_arrays_test["reconstructed"][var][pseudodata_pass_reco_test],
            simulation_event_arrays_test["reconstructed"][var][simulation_pass_reco_test],
            step1_weights[simulation_pass_reco_test],
            cfg["bins"], cfg["range"],
            cfg["xlabel"], plot_title,
            f"{flags.plot_directory}/{flags.MC_name}_{flags.pseudodata_name}_closure_iteration1_step1_{var}.png"
        )

        plot_unfolded(
            simulation_event_arrays_test["MC"][f"MC_{var}"][simulation_pass_reco_test],
            pseudodata_event_arrays_test["MC"][f"MC_{var}"][pseudodata_pass_reco_test],
            simulation_event_arrays_test["MC"][f"MC_{var}"][simulation_pass_reco_test],
            step2_weights[simulation_pass_reco_test],
            cfg["bins"], cfg["range"],
            cfg["xlabel"], plot_title,
            f"{flags.plot_directory}/{flags.MC_name}_{flags.pseudodata_name}_closure_iteration4_step2_{var}.png"
        )

if __name__ == "__main__":
    main()