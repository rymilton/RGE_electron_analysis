import argparse
import uproot
import glob
import awkward as ak
import numpy as np
import time
import h5py as h5
import os
from utils import LoadYaml, open_data

def parse_arguments():
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        "--input_file",
        default="/home/rmilton/work_dir/rge_datasets/job_9586_LD2Csolid_clasdis_deuteron_zh0_3k/ntuples/ntuples_LD2Csolid_clasdis_deuteron_100mil_zh0-9586-0.root",
        help="ROOT file containing tuples from tuple_maker",
        type=str,
    )
    parser.add_argument(
        "--output_directory",
        default="/home/rmilton/work_dir/rge_datasets/job_9586_LD2Csolid_clasdis_deuteron_zh0_3k/eventbuilder_electrons/",
        help="Directory to store event builder electrons",
        type=str,
    )
    parser.add_argument(
        "--output_file",
        default="electrons_eventbuilder_LD2Csolid_clasdis_deuteron_100mil_zh0-9586-0.root",
        help="ROOT file containing tuples from tuple_maker",
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

    flags = parser.parse_args()

    return flags

def save_output(
    events,
    output_directory,
    output_file,
    branches_to_save,
    save_MC = False,
    MC_branches_to_save = None
):
    reconstructed_dictionary = {}
    print("Saving reconstructed electrons")
    for field in branches_to_save:
        reconstructed_dictionary[field] = events["reconstructed"][field]
    if save_MC:
        MC_dictionary = {}
        for field in MC_branches_to_save:
            MC_dictionary[field] = events["MC"][field]
    
    os.makedirs(output_directory, exist_ok=True)
    full_output_path = os.path.join(output_directory, output_file)

    with uproot.recreate(full_output_path) as file:
        file["reconstructed_electrons"] = reconstructed_dictionary
        if save_MC:
            file["MC_electrons"] = MC_dictionary

def get_eventbuilder_electrons(events):
    status = events["reconstructed"]["status"]
    pid = events["reconstructed"]["pid"]
    trigger_electron_first_mask = (pid[:, 0] == 11) & (status[:, 0] < 0)
    events = events[trigger_electron_first_mask]

    trigger_electron_mask = (events["reconstructed"]["pid"] == 11) & (events["reconstructed"]["status"] <= -2000) & (events["reconstructed"]["status"] > -4000)
    events["reconstructed"] = events["reconstructed"][trigger_electron_mask]
    number_of_electrons = ak.num(events["reconstructed"]["pid"], axis = 1)
    # Removing events without any trigger electrons
    events = events[number_of_electrons > 0]
    if ak.any(number_of_electrons>1):
        raise ValueError("More than 1 trigger electron found in some events")
    
    events["reconstructed"] = events["reconstructed"][:,0]
    sampling_fraction = (events["reconstructed"]["E_PCAL"] + events["reconstructed"]["E_ECOUT"] + events["reconstructed"]["E_ECIN"]) / events["reconstructed"]["p"]
    events["reconstructed"] = ak.with_field(
        events["reconstructed"],
        sampling_fraction,
        "SF"
    )
    print(f"Have {len(events)} events after event builder electron cuts")
    return events
def calc_p(px, py, pz):
    return np.sqrt(px**2 + py**2 + pz**2)
def calc_theta_lab(px, py, pz):
    return np.arctan2(np.sqrt(px**2 + py**2), pz)
def calc_phi(px, py):
    return np.arctan2(py, px)
def calc_Q2(p, beam_E, theta):
    return 4 * p * beam_E* np.sin(theta/2)*np.sin(theta/2)
def calc_nu(p, beam_E):
    return beam_E - p
def calc_xb(Q2, beam_E, nu):
    proton_mass = .938
    return np.array(Q2/(2*proton_mass*nu))
def calc_y(p, beam_E):
    return calc_nu(p, beam_E)/beam_E
def calc_W2(p, beam_E, theta):
    proton_mass = .938
    return proton_mass*proton_mass + 2*proton_mass*calc_nu(p, beam_E) - calc_Q2(p, beam_E, theta)

def get_DIS_quantities(events):
    E_beam = 10.547
    electrons = events["reconstructed"]
    electrons["Q2"] = calc_Q2(electrons["p"], E_beam, electrons["theta"])
    electrons["nu"] = calc_nu(electrons["p"], E_beam)
    electrons["x"] = calc_xb(electrons["Q2"], E_beam, electrons["nu"])
    electrons["y"] = calc_y(electrons["p"], E_beam)
    electrons["W"] = np.sqrt(calc_W2(electrons["p"], E_beam, electrons["theta"]))
    
    electrons["theta_degrees"] = electrons["theta"]*180/np.pi
    electrons["phi_degrees"] = electrons["phi"]*180/np.pi

    events["reconstructed"] = electrons
    return events
def get_DIS_quantities_MC(events):
    E_beam = 10.547
    electrons = events["MC"]
    electrons["MC_p"] = calc_p(electrons["MC_px"], electrons["MC_py"], electrons["MC_pz"])
    electrons["MC_theta"] = calc_theta_lab(electrons["MC_px"], electrons["MC_py"], electrons["MC_pz"])
    electrons["MC_phi"] = calc_phi(electrons["MC_px"], electrons["MC_py"])
    electrons["MC_Q2"] = calc_Q2(electrons["MC_p"], E_beam, electrons["MC_theta"])
    electrons["MC_nu"] = calc_nu(electrons["MC_p"], E_beam)
    electrons["MC_x"] = calc_xb(electrons["MC_Q2"], E_beam, electrons["MC_nu"])
    electrons["MC_y"] = calc_y(electrons["MC_p"], E_beam)
    electrons["MC_W"] = np.sqrt(calc_W2(electrons["MC_p"], E_beam, electrons["MC_theta"]))
    electrons["MC_theta_degrees"] = electrons["MC_theta"]*180/np.pi
    electrons["MC_phi_degrees"] = electrons["MC_phi"]*180/np.pi

    events["MC"] = electrons
    return events
def get_MC_electrons(events):
    events["MC"] = events["MC"][events["MC"]["MC_pid"]==11]
    
    # For events with multiple electrons, only keeping the highest pz electron, which is the first
    # The secondary electrons all have low pz values -- below .5 GeV
    events["MC"] = events["MC"][:, 0]

    return events

def main():
    flags = parse_arguments()
    
    parameters = LoadYaml(flags.config, flags.config_directory)

    events_array = open_data(
        data_path = flags.input_file,
        branches_to_open = parameters["BRANCHES_TO_OPEN"],
        data_tree_name = "data",
        open_MC = flags.save_MC,
        MC_branches_to_open = parameters["MC_BRANCHES_TO_OPEN"] if flags.save_MC else None,
        MC_tree_name = "MC",
    )

    # Removing events with no reconstructed particles
    events_array = events_array[ak.num(events_array["reconstructed"]["pid"], axis=1)>0]
    print(f"Have {len(events_array)} events after removing empty events")
    events_array = get_eventbuilder_electrons(events_array)
    events_array = get_DIS_quantities(events_array)

    if flags.save_MC:
        events_array = get_MC_electrons(events_array)
        events_array = get_DIS_quantities_MC(events_array)

    save_output(
        events_array,
        flags.output_directory,
        flags.output_file,
        parameters["BRANCHES_TO_SAVE"],
        flags.save_MC,
        parameters["MC_BRANCHES_TO_SAVE"] if flags.save_MC else None)
    
if __name__ == "__main__":
    main()