import ROOT
import numpy as np
import uproot as ur
import awkward as ak
import os
import matplotlib.pyplot as plt
import mplhep as hep
hep.style.use("CMS")
hep.style.use(hep.style.CMS)
import matplotlib.colors as mcolors
from matplotlib import colormaps
from scipy.optimize import curve_fit
import pandas as pd
from mpl_toolkits.axes_grid1 import make_axes_locatable

import sys
sys.path.append('/home/ryan/unbinned_unfolding_october2025/build/RooUnfold/') # Insert path to your build directory
from omnifold import OmniFold_helper_functions

from cycler import cycler
plt.rcParams["axes.prop_cycle"] = cycler('color', ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'])

load_trained_omnifold_model = False
total_number_of_events = 5000000
MC_for_unfolding = "GiBUU"  # GiBUU or clasdis
num_sectors = 6
use_unfolding = False
E_beam = 10.547
include_data = True

def DivideWithErrors(numerator, numerator_error, dividend, dividend_error):
    quotient = numerator/dividend
    error = quotient * np.sqrt((numerator_error/numerator)**2 + (dividend_error/dividend)**2)
    return quotient, error
def np_to_TVector(array):
    vector = ROOT.TVector(len(array))
    for i, entry in enumerate(array):
        vector[i] = entry
    return vector
def TVector_to_np(vector):
    out_array = []
    for i in range(vector.GetNoElements()):
        out_array.append(vector[i])
    return np.array(out_array)

plot_title = f"RGE 020131-20176 pass 0.9: C/LD2"

MC_BRANCHES_TO_LOAD = [
    "MC_pid",
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
    "pid",
    "p_x",
    "p_y",
    "p_z",
    "p",
    "status",
    "theta_degrees",
    "phi_degrees",
    "v_x",
    "v_y",
    "v_z",
    "theta",
    "phi",
    "sector",
    "NDF",
    "chi2",
    "E_PCAL",
    "E_ECIN",
    "E_ECOUT",
    "PCAL_U",
    "PCAL_V",
    "PCAL_W",
    "DC_region1_x",
    "DC_region1_y",
    "DC_region1_z",
    "DC_region1_edge",
    "DC_region2_x",
    "DC_region2_y",
    "DC_region2_z",
    "DC_region2_edge",
    "DC_region3_x",
    "DC_region3_y",
    "DC_region3_z",
    "DC_region3_edge",
    "Q2",
    "nu",
    "x",
    "y",
    "W",
    "pass_reco"
]

separated_sim_liquid_and_solid = True


gibuu_directories = {"MC1_liquid" : "/home/ryan/", "MC1_solid" : "/home/ryan/"}
gibuu_files = {"MC1_liquid" : "candidate_electrons_LD2Cliquid_gibuu_deuteron_3000files_passreco.root", "MC1_solid" : "candidate_electrons_LD2Csolid_gibuu_carbon_3000files_passreco.root"}
gibuu_event_dictionary = {"reconstructed_liquid": ak.Array([]), "MC_liquid": ak.Array([]), "reconstructed_solid": ak.Array([]), "MC_solid": ak.Array([])}

clasdis_directories = {"MC2_liquid" : "/home/ryan/", "MC2_solid" : "/home/ryan/"}
clasdis_files = {"MC2_liquid" : "candidate_electrons_L2DCliquid_clasdis_deuteron_zh0_3000files_passreco.root", "MC2_solid" : "candidate_electrons_L2DCsolid_clasdis_deuteron_zh0_3000files_passreco.root"}
clasdis_event_dictionary = {"reconstructed_liquid": ak.Array([]), "MC_liquid": ak.Array([]), "reconstructed_solid": ak.Array([]), "MC_solid": ak.Array([])}

if MC_for_unfolding == "GiBUU":
    MC1_directories = gibuu_directories
    MC1_files = gibuu_files
    MC1_event_dictionary = gibuu_files
    MC1_name = "GiBUU"
    MC2_directories = clasdis_directories
    MC2_files = clasdis_files
    MC2_event_dictionary = clasdis_event_dictionary
    MC2_name = "clasdis"
else:
    MC2_directories = gibuu_directories
    MC2_files = gibuu_files
    MC2_event_dictionary = gibuu_files
    MC2_name = "GiBUU"
    MC1_directories = clasdis_directories
    MC1_files = clasdis_files
    MC1_event_dictionary = clasdis_event_dictionary
    MC1_name = "clasdis"

data_directory = "/media/miguel/Elements_2024/CLAS_data/C/electron_candidates/"
data_file = "candidate_electrons_020131-020176_pass09.root"

MC_names = {"MC1": MC1_name, "MC2": MC2_name}
# Opening MC 1
for name in MC1_directories:
    print(os.path.join(MC1_directories[name], MC1_files[name]))
    with ur.open(f"{os.path.join(MC1_directories[name], MC1_files[name])}:reconstructed_electrons") as events:
        reco_electrons = events.arrays(filter_name = BRANCHES_TO_LOAD)
        if "solid" in name:
            MC1_event_dictionary["reconstructed_solid"] = reco_electrons
        else:
            MC1_event_dictionary["reconstructed_liquid"] = reco_electrons
    with ur.open(f"{os.path.join(MC1_directories[name], MC1_files[name])}:MC_electrons") as events:
        truth_electrons = events.arrays(filter_name = MC_BRANCHES_TO_LOAD)
        if "solid" in name:
            MC1_event_dictionary["MC_solid"] = truth_electrons
        else:
            MC1_event_dictionary["MC_liquid"] = truth_electrons
# Opening MC 2
for name in MC2_directories:
    print(os.path.join(MC2_directories[name], MC2_files[name]))
    with ur.open(f"{os.path.join(MC2_directories[name], MC2_files[name])}:reconstructed_electrons") as events:
        reco_electrons = events.arrays(filter_name = BRANCHES_TO_LOAD)
        if "solid" in name:
            MC2_event_dictionary["reconstructed_solid"] = reco_electrons
        else:
            MC2_event_dictionary["reconstructed_liquid"] = reco_electrons
    with ur.open(f"{os.path.join(MC2_directories[name], MC2_files[name])}:MC_electrons") as events:
        truth_electrons = events.arrays(filter_name = MC_BRANCHES_TO_LOAD)
        if "solid" in name:
            MC2_event_dictionary["MC_solid"] = truth_electrons
        else:
            MC2_event_dictionary["MC_liquid"] = truth_electrons

print(os.path.join(data_directory, data_file))
data_event_arrays = {"reconstructed":ak.Array([])}
with ur.open(f"{os.path.join(data_directory, data_file)}:reconstructed_electrons") as events:
    data_event_arrays["reconstructed"] = events.arrays(filter_name = BRANCHES_TO_LOAD)

MC1_event_dictionary["reconstructed_solid"]["weight"] = 1
MC1_event_dictionary["reconstructed_liquid"]["weight"] = 1
MC1_event_dictionary["MC_solid"]["MC_p"] = np.sqrt(MC1_event_dictionary["MC_solid"]["MC_px"]**2+MC1_event_dictionary["MC_solid"]["MC_py"]**2+MC1_event_dictionary["MC_solid"]["MC_pz"]**2)
MC1_event_dictionary["MC_liquid"]["MC_p"] = np.sqrt(MC1_event_dictionary["MC_liquid"]["MC_px"]**2+MC1_event_dictionary["MC_liquid"]["MC_py"]**2+MC1_event_dictionary["MC_liquid"]["MC_pz"]**2)

MC1_event_arrays = {}
MC1_event_arrays["reconstructed"] = ak.concatenate([MC1_event_dictionary["reconstructed_solid"], MC1_event_dictionary["reconstructed_liquid"]])
MC1_event_arrays["MC"] = ak.concatenate([MC1_event_dictionary["MC_solid"], MC1_event_dictionary["MC_liquid"]])
MC1_event_arrays["reconstructed"]["theta_degrees"] = MC1_event_arrays["reconstructed"]["theta_degrees"]
MC1_event_arrays["MC"]["MC_theta_degrees"] = MC1_event_arrays["MC"]["MC_theta_degrees"]
MC1_event_arrays["reconstructed"]["phi_degrees"] = MC1_event_arrays["reconstructed"]["phi_degrees"]
MC1_event_arrays["MC"]["MC_phi_degrees"] = MC1_event_arrays["MC"]["MC_phi_degrees"]
MC1_event_arrays["reconstructed"]["vz"] = MC1_event_arrays["reconstructed"]["v_z"]
MC1_event_arrays = ak.Array(MC1_event_arrays)

permutation = np.random.permutation(len(MC1_event_arrays))
MC1_branches_shuffled = MC1_event_arrays[permutation]
MC1_event_arrays = MC1_branches_shuffled[:total_number_of_events]
num_MC1_train_events = int(.8*len(MC1_event_arrays))
MC1_event_arrays_train = MC1_event_arrays[:num_MC1_train_events]
MC1_event_arrays_test = MC1_event_arrays[num_MC1_train_events:]

MC1_pass_truth_train = (MC1_event_arrays_train["MC"]["MC_p"]>2) & (MC1_event_arrays_train["MC"]["MC_p"]<8) & (MC1_event_arrays_train["MC"]["MC_W"]>2) & (MC1_event_arrays_train["MC"]["MC_y"]<0.8) & (MC1_event_arrays_train["MC"]["MC_theta_degrees"]>5)
MC1_pass_reco_train = MC1_event_arrays_train["reconstructed"]["pass_reco"]
MC1_pass_truth_test = (MC1_event_arrays_test["MC"]["MC_p"]>2) & (MC1_event_arrays_test["MC"]["MC_p"]<8) & (MC1_event_arrays_test["MC"]["MC_W"]>2) & (MC1_event_arrays_test["MC"]["MC_y"]<0.8) & (MC1_event_arrays_test["MC"]["MC_theta_degrees"]>5)
MC1_pass_reco_test = MC1_event_arrays_test["reconstructed"]["pass_reco"]


MC2_event_dictionary["reconstructed_solid"]["weight"] = 1
MC2_event_dictionary["reconstructed_liquid"]["weight"] = 1
MC2_event_dictionary["MC_solid"]["MC_p"] = np.sqrt(MC2_event_dictionary["MC_solid"]["MC_px"]**2+MC2_event_dictionary["MC_solid"]["MC_py"]**2+MC2_event_dictionary["MC_solid"]["MC_pz"]**2)
MC2_event_dictionary["MC_liquid"]["MC_p"] = np.sqrt(MC2_event_dictionary["MC_liquid"]["MC_px"]**2+MC2_event_dictionary["MC_liquid"]["MC_py"]**2+MC2_event_dictionary["MC_liquid"]["MC_pz"]**2)
MC2_event_arrays = {}
MC2_event_arrays["reconstructed"] = ak.concatenate([MC2_event_dictionary["reconstructed_solid"], MC2_event_dictionary["reconstructed_liquid"]])
MC2_event_arrays["MC"] = ak.concatenate([MC2_event_dictionary["MC_solid"], MC2_event_dictionary["MC_liquid"]])
MC2_event_arrays["reconstructed"]["theta_degrees"] = MC2_event_arrays["reconstructed"]["theta_degrees"]
MC2_event_arrays["MC"]["MC_theta_degrees"] = MC2_event_arrays["MC"]["MC_theta_degrees"]
MC2_event_arrays["reconstructed"]["phi_degrees"] = MC2_event_arrays["reconstructed"]["phi_degrees"]
MC2_event_arrays["MC"]["MC_phi_degrees"] = MC2_event_arrays["MC"]["MC_phi_degrees"]
MC2_event_arrays["reconstructed"]["vz"] = MC2_event_arrays["reconstructed"]["v_z"]
MC2_event_arrays = ak.Array(MC2_event_arrays)

permutation = np.random.permutation(len(MC2_event_arrays))
MC2_branches_shuffled = MC2_event_arrays[permutation]
MC2_event_arrays = MC2_branches_shuffled[:total_number_of_events]
num_MC2_train_events = int(.8*len(MC2_event_arrays))
MC2_event_arrays_train = MC2_event_arrays[:num_MC2_train_events]
MC2_event_arrays_test = MC2_event_arrays[num_MC2_train_events:]

MC2_pass_truth_train = (MC2_event_arrays_train["MC"]["MC_p"]>2) & (MC2_event_arrays_train["MC"]["MC_p"]<8) & (MC2_event_arrays_train["MC"]["MC_W"]>2) & (MC2_event_arrays_train["MC"]["MC_y"]<0.8) & (MC2_event_arrays_train["MC"]["MC_theta_degrees"]>5)
MC2_pass_reco_train = MC2_event_arrays_train["reconstructed"]["pass_reco"]
MC2_pass_truth_test = (MC2_event_arrays_test["MC"]["MC_p"]>2) & (MC2_event_arrays_test["MC"]["MC_p"]<8) & (MC2_event_arrays_test["MC"]["MC_W"]>2) & (MC2_event_arrays_test["MC"]["MC_y"]<0.8) & (MC2_event_arrays_test["MC"]["MC_theta_degrees"]>5)
MC2_pass_reco_test = MC2_event_arrays_test["reconstructed"]["pass_reco"]

data_event_arrays["reconstructed"]["vz"] = data_event_arrays["reconstructed"]["v_z"]
data_event_arrays = ak.Array(data_event_arrays)
permutation = np.random.permutation(len(data_event_arrays))
data_shuffled = data_event_arrays[permutation]
data_event_arrays = data_shuffled[:total_number_of_events]
num_data_train_events = int(.8*len(MC1_event_arrays))
data_event_arrays_train = data_event_arrays[:num_data_train_events]
data_event_arrays_test = data_event_arrays[num_data_train_events:]

data_pass_reco_train = data_event_arrays_train["reconstructed"]["pass_reco"]
data_pass_reco_test = data_event_arrays_test["reconstructed"]["pass_reco"]

variables_to_unfold = ["p", "Q2", "x", "phi_degrees", "theta_degrees"]

sim_MCreco_dict_train, sim_MCgen_dict_train, data_dict_train = {}, {}, {}
for variable in variables_to_unfold:
    sim_MCreco_dict_train[variable] = np.array(MC1_event_arrays_train["reconstructed"][variable])
    sim_MCgen_dict_train[variable] = np.array(MC1_event_arrays_train["MC"]["MC_"+variable])
    data_dict_train[variable] = np.array(data_event_arrays_train["reconstructed"][variable])
df_MCgen_train = ROOT.RDF.FromNumpy(sim_MCgen_dict_train)
df_MCreco_train = ROOT.RDF.FromNumpy(sim_MCreco_dict_train)
df_measured_train = ROOT.RDF.FromNumpy(data_dict_train)
sim_pass_reco_vector_train = np_to_TVector(MC1_pass_reco_train)
data_pass_reco_vector_train = np_to_TVector(data_pass_reco_train)

if not load_trained_omnifold_model:
    unbinned_unfolding = ROOT.RooUnfoldOmnifold()
    unbinned_unfolding.SetSaveDirectory("/home/ryan/clas_analysis/clas12-rge-analysis/analysis/")
    unbinned_unfolding.SetModelSaveName(f"RGE_{MC_names['MC1']}_allobservables_nopasstruth_withpassreco_traintestsplit_5million")
    unbinned_unfolding.SetMCgenDataFrame(df_MCgen_train)
    unbinned_unfolding.SetMCrecoDataFrame(df_MCreco_train)
    unbinned_unfolding.SetMCPassReco(sim_pass_reco_vector_train)
    unbinned_unfolding.SetMeasuredDataFrame(df_measured_train)
    unbinned_unfolding.SetMeasuredPassReco(data_pass_reco_vector_train)
    unbinned_unfolding.SetNumIterations(4)
    unbinned_results = unbinned_unfolding.UnbinnedOmnifold()

sim_MCreco_dict_test, sim_MCgen_dict_test, data_dict_test = {}, {}, {}
for variable in variables_to_unfold:
    sim_MCreco_dict_test[variable] = np.array(MC1_event_arrays_test["reconstructed"][variable])
    sim_MCgen_dict_test[variable] = np.array(MC1_event_arrays_test["MC"]["MC_"+variable])
    data_dict_test[variable] = np.array(data_event_arrays_test["reconstructed"][variable])
df_MCgen_test = ROOT.RDF.FromNumpy(sim_MCgen_dict_test)
df_MCreco_test = ROOT.RDF.FromNumpy(sim_MCreco_dict_test)
sim_pass_reco_vector_test = np_to_TVector(MC1_pass_reco_test)

unbinned_unfolding = ROOT.RooUnfoldOmnifold()
unbinned_unfolding.SetTestMCgenDataFrame(df_MCgen_test)
unbinned_unfolding.SetTestMCrecoDataFrame(df_MCreco_test)
unbinned_unfolding.SetTestMCPassReco(sim_pass_reco_vector_test)
unbinned_unfolding.SetLoadModelPath(f"/home/ryan/clas_analysis/clas12-rge-analysis/analysis/RGE_{MC_names['MC1']}_allobservables_nopasstruth_withpassreco_traintestsplit_5million_iteration_0.pkl")
test_unbinned_results = unbinned_unfolding.TestUnbinnedOmnifold()
step1_weights = TVector_to_np(ROOT.std.get[0](test_unbinned_results))
plt.figure()
plt.hist(step1_weights, bins=100)
plt.xlabel("Step 1 weights before testing")
plt.savefig(f"./RGE_{MC_names['MC1']}_iteration1_step1weights.png")
print("Min step 1 weight: ", np.min(step1_weights))
unbinned_unfolding.SetTestMCgenDataFrame(df_MCgen_test)
unbinned_unfolding.SetTestMCrecoDataFrame(df_MCreco_test)
unbinned_unfolding.SetTestMCPassReco(sim_pass_reco_vector_test)
unbinned_unfolding.SetLoadModelPath(f"/home/ryan/clas_analysis/clas12-rge-analysis/analysis/RGE_{MC_names['MC1']}_allobservables_nopasstruth_withpassreco_traintestsplit_5million_iteration_3.pkl")
test_unbinned_results = unbinned_unfolding.TestUnbinnedOmnifold()
step2_weights = TVector_to_np(ROOT.std.get[1](test_unbinned_results))
plt.figure()
plt.hist(step2_weights, bins=100)
plt.xlabel("Step 2 weights after testing")
plt.savefig(f"./RGE_{MC_names['MC1']}_iteration4_step2weights.png")



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
    settings = variable_settings[var]
    num_bins = settings["bins"]
    low_bin, high_bin = settings["range"]
    xlabel = settings["xlabel"]
    
    # ---------- RECONSTRUCTED PLOTS ----------
    fig, axs = plt.subplots(2, 1, figsize=(10, 10), sharex=True, gridspec_kw={
        'height_ratios': [2, 1],   # top panel bigger
        'hspace': 0.05             # reduce space between panels
    })
    axs = axs.flatten()

    MC1_reco_counts, bin_edges = np.histogram(
        MC1_event_arrays_test["reconstructed"][var][MC1_pass_reco_test],
        bins=num_bins,
        range=(low_bin, high_bin),
    )
    MC1_reco_errors = np.sqrt(MC1_reco_counts)

    bin_widths = np.diff(bin_edges)
    bin_centers = 0.5*(bin_edges[:-1] + bin_edges[1:])

    total_MC1 = np.sum(MC1_reco_counts)
    norm_MC1_counts = MC1_reco_counts / (total_MC1 * bin_widths)
    norm_MC1_errors = MC1_reco_errors / (total_MC1 * bin_widths)

    axs[0].hist(
        MC1_event_arrays_test["reconstructed"][var][MC1_pass_reco_test],
        bins=num_bins,
        range=(low_bin, high_bin),
        density=True,
        label=f"{MC_names['MC1']} Reco",
        histtype="step",
        color="#2ca02c",
        linewidth=2,
    )

    data_counts, bin_edges = np.histogram(
        data_event_arrays_test["reconstructed"][var][data_pass_reco_test],
        bins=num_bins,
        range=(low_bin, high_bin),
    )
    data_errors = np.sqrt(data_counts)

    total_data = np.sum(data_counts)
    norm_data_counts = data_counts / (total_data * bin_widths)
    norm_data_errors = data_errors / (total_data * bin_widths)

    axs[0].hist(
        data_event_arrays_test["reconstructed"][var][data_pass_reco_test],
        bins=num_bins,
        range=(low_bin, high_bin),
        density=True,
        label=f"RGE Reco",
        histtype="step",
        color="#ff7f0e",
        linewidth=2,
    )

    # Apply unfolding weights (step1)
    unfolded_counts, _ = np.histogram(
        MC1_event_arrays_test["reconstructed"][var][MC1_pass_reco_test],
        weights=step1_weights[MC1_pass_reco_test],
        bins=num_bins,
        range=(low_bin, high_bin),
    )
    unfolded_errors = np.sqrt(unfolded_counts)

    total_unf = np.sum(unfolded_counts)
    norm_counts = unfolded_counts / (total_unf * bin_widths)
    norm_errors = unfolded_errors / (total_unf * bin_widths)

    axs[0].errorbar(
        bin_centers,
        norm_counts,
        yerr=norm_errors,
        fmt="o",
        color="#1f77b4",
        label=f"{MC_names['MC1']} with Step 1 weights",
        markersize=7,
    )

    axs[0].set_ylabel("Normalized entries")
    axs[0].legend(loc="upper right")

    MC1_ratio, MC1_ratio_error = DivideWithErrors(
        norm_MC1_counts,
        norm_MC1_errors,
        norm_counts,
        norm_errors
    )

    data_ratio, data_ratio_error = DivideWithErrors(
        norm_data_counts,
        norm_data_errors,
        norm_counts,
        norm_errors
    )

    axs[1].errorbar(
        bin_centers,
        MC1_ratio,
        yerr=MC1_ratio_error,
        fmt="o",
        color="#2ca02c",
        label=f"{MC_names['MC1']}",
        markersize=7,
    )

    axs[1].errorbar(
        bin_centers,
        data_ratio,
        yerr=data_ratio_error,
        fmt="o",
        color="#ff7f0e",
        label="RGE",
        markersize=7,
    )
    axs[1].axhline(1.0, color="red", linestyle="--", linewidth=1.5)
    axs[1].set_ylabel("Counts/Unfolded")
    axs[1].set_xlabel(xlabel)
    axs[1].set_ylim(0.5, 1.5)
    
    plt.tight_layout()
    plt.suptitle(plot_title)
    plt.savefig(f"./RGE_{MC_names['MC1']}_iteration1_{var}_traintestsplit_5million_withratios.png")
    plt.show()

    # ---------- TRUTH-LEVEL PLOTS ----------
    fig, axs = plt.subplots(2, 1, figsize=(10, 10), sharex=True, gridspec_kw={
        'height_ratios': [2, 1],   # top panel bigger
        'hspace': 0.05             # reduce space between panels
    })
    axs = axs.flatten()

    MC1_truth_counts, bin_edges = np.histogram(
        MC1_event_arrays_test["MC"][f"MC_{var}"][MC1_pass_truth_test],
        bins=num_bins,
        range=(low_bin, high_bin),
    )
    MC1_truth_errors = np.sqrt(MC1_truth_counts)

    bin_widths = np.diff(bin_edges)
    bin_centers = 0.5*(bin_edges[:-1] + bin_edges[1:])

    total_MC1_truth = np.sum(MC1_truth_counts)
    norm_MC1_truth_counts = MC1_truth_counts / (total_MC1_truth * bin_widths)
    norm_MC1_truth_errors = MC1_truth_errors / (total_MC1_truth * bin_widths)

    axs[0].hist(
        MC1_event_arrays_test["MC"][f"MC_{var}"][MC1_pass_truth_test],
        bins=num_bins,
        range=(low_bin, high_bin),
        density=True,
        label=f"{MC_names['MC1']} Truth",
        histtype="step",
        color="#2ca02c",
        linewidth=2,
    )

    MC2_truth_counts, bin_edges = np.histogram(
        MC2_event_arrays_test["MC"][f"MC_{var}"][MC2_pass_truth_test],
        bins=num_bins,
        range=(low_bin, high_bin),
    )
    MC2_truth_errors = np.sqrt(MC2_truth_counts)

    bin_widths = np.diff(bin_edges)
    bin_centers = 0.5*(bin_edges[:-1] + bin_edges[1:])

    total_MC2_truth = np.sum(MC2_truth_counts)
    norm_MC2_truth_counts = MC2_truth_counts / (total_MC2_truth * bin_widths)
    norm_MC2_truth_errors = MC2_truth_errors / (total_MC2_truth * bin_widths)

    axs[0].hist(
        MC2_event_arrays_test["MC"][f"MC_{var}"][MC2_pass_truth_test],
        bins=num_bins,
        range=(low_bin, high_bin),
        density=True,
        label=f"{MC_names['MC2']} Truth",
        histtype="step",
        color="#ff7f0e",
        linewidth=2,
    )

     # Apply unfolding weights (step1)
    unfolded_counts, _ = np.histogram(
        MC1_event_arrays_test["MC"][f"MC_{var}"][MC1_pass_truth_test],
        weights=step2_weights[MC1_pass_truth_test],
        bins=num_bins,
        range=(low_bin, high_bin),
    )
    unfolded_errors = np.sqrt(unfolded_counts)

    total_unf = np.sum(unfolded_counts)
    norm_counts = unfolded_counts / (total_unf * bin_widths)
    norm_errors = unfolded_errors / (total_unf * bin_widths)

    axs[0].errorbar(
        bin_centers,
        norm_counts,
        yerr=norm_errors,
        fmt="o",
        color="#1f77b4",
        label=f"RGE Unfolded",
        markersize=7,
    )

    axs[0].set_ylabel("Normalized entries")
    axs[0].legend(loc="upper right")

    MC1_truth_ratio, MC1_truth_ratio_error = DivideWithErrors(
        norm_MC1_truth_counts,
        norm_MC1_truth_errors,
        norm_counts,
        norm_errors
    )

    MC2_truth_ratio, MC2_truth_ratio_error = DivideWithErrors(
        norm_MC2_truth_counts,
        norm_MC2_truth_errors,
        norm_counts,
        norm_errors
    )

    axs[1].errorbar(
        bin_centers,
        MC1_truth_ratio,
        yerr=MC1_truth_ratio_error,
        fmt="o",
        color="#2ca02c",
        label=f"{MC_names['MC1']}",
        markersize=7,
    )

    axs[1].errorbar(
        bin_centers,
        MC2_truth_ratio,
        yerr=MC2_truth_ratio_error,
        fmt="o",
        color="#ff7f0e",
        label=f"{MC_names['MC2']}",
        markersize=7,
    )
    axs[1].axhline(1.0, color="red", linestyle="--", linewidth=1.5)
    axs[1].set_ylabel("Counts/Unfolded")
    axs[1].set_xlabel(xlabel)
    axs[1].set_ylim(0.5, 1.5)

    plt.tight_layout()
    plt.suptitle(plot_title)
    plt.savefig(f"RGE_{MC_names['MC1']}_iteration4_{var}_traintestsplit_5million_withratios.png")
    plt.show()
