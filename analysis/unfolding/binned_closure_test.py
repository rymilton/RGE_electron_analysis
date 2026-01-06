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

from cycler import cycler
plt.rcParams["axes.prop_cycle"] = cycler('color', ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'])
import sys
sys.path.append('/home/ryan/unbinned_unfolding_october2025/build/RooUnfold/') # Insert path to your build directory
from omnifold import OmniFold_helper_functions

load_trained_omnifold_model = True
total_number_of_events = 1000000
num_sectors = 6
use_unfolding = False
E_beam = 10.547
include_data = True

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

MC1_name = "clasdis"
MC2_name = "GiBUU"
plot_title = f"{MC1_name} vs. {MC2_name}: C/LD2"

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
# NOTE: MC1 is the simulation data, MC2 is the pseudodata
MC1_directories = {"MC1_liquid" : "/home/ryan/", "MC1_solid" : "/home/ryan/"}
MC1_files = {"MC1_liquid" : "candidate_electrons_LD2Cliquid_gibuu_deuteron_3000files_passreco.root", "MC1_solid" : "candidate_electrons_LD2Csolid_gibuu_carbon_3000files_passreco.root"}
MC1_event_dictionary = {"reconstructed_liquid": ak.Array([]), "MC_liquid": ak.Array([]), "reconstructed_solid": ak.Array([]), "MC_solid": ak.Array([])}

MC2_directories = {"MC2_liquid" : "/home/ryan/", "MC2_solid" : "/home/ryan/"}
MC2_files = {"MC2_liquid" : "candidate_electrons_L2DCliquid_clasdis_deuteron_zh0_3000files_passreco.root", "MC2_solid" : "candidate_electrons_L2DCsolid_clasdis_deuteron_zh0_3000files_passreco.root"}
MC2_event_dictionary = {"reconstructed_liquid": ak.Array([]), "MC_liquid": ak.Array([]), "reconstructed_solid": ak.Array([]), "MC_solid": ak.Array([])}

MC_names = {"MC1": "GiBUU", "MC2": "clasdis"}
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
MC1_event_arrays = ak.Array(MC1_event_arrays)


permutation = np.random.permutation(len(MC1_event_arrays))
MC1_branches_shuffled = MC1_event_arrays[permutation]
MC1_event_arrays = MC1_branches_shuffled[:total_number_of_events]
MC1_pass_truth = (MC1_event_arrays["MC"]["MC_p"]>2) & (MC1_event_arrays["MC"]["MC_p"]<8) & (MC1_event_arrays["MC"]["MC_W"]>2) & (MC1_event_arrays["MC"]["MC_y"]<0.8) & (MC1_event_arrays["MC"]["MC_theta_degrees"]>5)

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
MC2_event_arrays = ak.Array(MC2_event_arrays)


permutation = np.random.permutation(len(MC2_event_arrays))
MC2_branches_shuffled = MC2_event_arrays[permutation]
MC2_event_arrays = MC2_branches_shuffled[:total_number_of_events]
MC2_pass_truth = (MC2_event_arrays["MC"]["MC_p"]>2) & (MC2_event_arrays["MC"]["MC_p"]<8) & (MC2_event_arrays["MC"]["MC_W"]>2) & (MC2_event_arrays["MC"]["MC_y"]<0.8) & (MC2_event_arrays["MC"]["MC_theta_degrees"]>5)

variable_settings = {
    "p":              {"bins": 50, "range": (0, 10), "xlabel": "p (GeV)"},
    "theta_degrees":  {"bins": 50, "range": (0, 60), "xlabel": "#theta (deg.)"},
    "phi_degrees":    {"bins": 50, "range": (-180, 180), "xlabel": "#phi (deg.)"},
    "x":              {"bins": 50, "range": (0, 1), "xlabel": "x"},
    "Q2":             {"bins": 50, "range": (0, 10), "xlabel": "Q^{2} (GeV^{2})"},
}

variables_to_unfold = ["p", "theta_degrees", "phi_degrees", "x", "Q2"]

for var in variables_to_unfold:
    print(f"Binning {var}")
    MCgen_hist    = ROOT.TH1D(f"MCgen_hist_{var}", f"MCgen_hist_{var}", variable_settings[var]["bins"], variable_settings[var]["range"][0], variable_settings[var]["range"][1])
    MCreco_hist   = ROOT.TH1D(f"MCreco_hist_{var}", f"MCreco_hist_{var}", variable_settings[var]["bins"], variable_settings[var]["range"][0], variable_settings[var]["range"][1])
    response      = ROOT.RooUnfoldResponse(variable_settings[var]["bins"], variable_settings[var]["range"][0], variable_settings[var]["range"][1], variable_settings[var]["bins"], variable_settings[var]["range"][0], variable_settings[var]["range"][1])
    truth_hist    = ROOT.TH1D (f"truth_hist_{var}", f"truth_hist_{var}", variable_settings[var]["bins"], variable_settings[var]["range"][0], variable_settings[var]["range"][1])
    measured_hist = ROOT.TH1D (f"measured_hist_{var}", f"measured_hist_{var}", variable_settings[var]["bins"], variable_settings[var]["range"][0], variable_settings[var]["range"][1])

    print(f"Binning MC1 {var}")
    # Populating simulation histograms
    for (MCgen, MCreco, pass_reco, pass_truth) in zip(MC1_event_arrays["MC"][f"MC_{var}"], MC1_event_arrays["reconstructed"][f"{var}"], MC1_event_arrays["reconstructed"]["pass_reco"], MC1_pass_truth):
        if not pass_truth:
            continue
        MCgen_hist.Fill(MCgen)
        if pass_reco:
            response.Fill(MCreco, MCgen)
            MCreco_hist.Fill(MCreco)
        else:
            response.Miss(MCgen)
    print(f"Binning MC2 {var}")
    for (MCgen, MCreco, pass_reco, pass_truth) in zip(MC2_event_arrays["MC"][f"MC_{var}"], MC2_event_arrays["reconstructed"][f"{var}"], MC2_event_arrays["reconstructed"]["pass_reco"], MC2_pass_truth):
        if not pass_truth:
            continue
        truth_hist.Fill(MCgen)
        if pass_reco:
            measured_hist.Fill(MCreco)
    print(f"Unfolding {var}")
    binned_unfolding = ROOT.RooUnfoldOmnifold(response, measured_hist, 4)
    binned_unfolding_hist = binned_unfolding.Hunfold()
    
    print("Done unfolding")
    ROOT.gROOT.SetBatch(True)
    c_binned_unfolding = ROOT.TCanvas()
    binned_unfolding_hist.SetStats(0)
    binned_unfolding_hist.SetTitle("clasdis vs. GiBUU C/LD2: Binned UniFold 4 iterations")
    # binned_unfolding_hist.SetTitleSize(0.03)  # default is ~0.05-0.06
    binned_unfolding_hist.SetLineColor(3)
    truth_hist.SetLineColor(2)

    binned_unfolding_hist.GetXaxis().SetTitle(variable_settings[var]["xlabel"])
    binned_unfolding_hist.GetYaxis().SetTitle("Normalized entries")

    #Normalizing the histograms
    print("Normalizing hists")
    def normalize_density(h):
        # Convert bin contents to densities by dividing by bin width
        for i in range(1, h.GetNbinsX()+1):
            width = h.GetBinWidth(i)
            if width > 0:
                h.SetBinContent(i, h.GetBinContent(i) / width)

        # Now scale so area = 1 (integral over width)
        I = h.Integral("width")
        if I > 0:
            h.Scale(1.0 / I)
        else:
            print(f"WARNING: histogram {h.GetName()} has zero integral — skipping density norm.")

    # Apply to all your histograms
    normalize_density(truth_hist)
    normalize_density(MCgen_hist)
    normalize_density(binned_unfolding_hist)

    max_val = max(
    truth_hist.GetMaximum(),
    MCgen_hist.GetMaximum(),
    binned_unfolding_hist.GetMaximum()
    )

    # Add a bit of padding on top so lines don't touch the edge
    ymax = max_val * 1.1   # 20% padding

    # Set the max on the histogram that will be drawn first
    binned_unfolding_hist.SetMaximum(ymax)

    print("Drawing hists")
    binned_unfolding_hist.Draw()
    truth_hist.Draw("same hist")
    MCgen_hist.Draw("same hist")

    if var=="p":
        leg_binned_unfolding = ROOT.TLegend(0.7, 0.7, 0.9, 0.9)
    else:
        leg_binned_unfolding = ROOT.TLegend(0.6, 0.6, 0.8, 0.8)
    leg_binned_unfolding.AddEntry(truth_hist, "clasdis Truth","pl")
    leg_binned_unfolding.AddEntry(MCgen_hist, "GiBUU Truth", "pl")
    leg_binned_unfolding.AddEntry(binned_unfolding_hist, "clasdis Unfolded")
    leg_binned_unfolding.SetBorderSize(0)
    leg_binned_unfolding.SetFillStyle(0)
    leg_binned_unfolding.Draw()
    c_binned_unfolding.Draw()
    print("Saving hists")
    c_binned_unfolding.SaveAs(f"./binnedomnifold_1million_{var}_passtruth.pdf")


    
