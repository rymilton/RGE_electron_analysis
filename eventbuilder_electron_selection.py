import argparse
import awkward as ak
import numpy as np
import os
from utils import LoadYaml, open_data, save_output

def parse_arguments():
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        "--input_file",
        default="/home/rmilton/work_dir/rge_datasets/job_9586_LD2Csolid_clasdis_deuteron_zh0_3k/ntuples_LD2Csolid_clasdis_deuteron_zh0_3000files.root",
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
        default="electrons_eventbuilder_LD2Csolid_clasdis_deuteron_zh0_3000files.root",
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


def get_eventbuilder_electrons(events):
    # 1. Create a mask at the particle level (avoids pid[:, 0] crashes)
    pid = events["reconstructed"]["pid"]
    status = events["reconstructed"]["status"]

    trigger_electron_mask = (pid == 11) & (status <= -2000) & (status > -4000)

    # 2. Filter particles, keeping the jagged event structure intact
    jagged_electrons = events["reconstructed"][trigger_electron_mask]

    # 3. Count trigger electrons per event to build the flag
    number_of_electrons = ak.sum(trigger_electron_mask, axis=1)
    pass_trigger = number_of_electrons > 0

    if ak.any(number_of_electrons > 1):
        raise ValueError("More than 1 trigger electron found in some events")

    # 4. Flatten EVERY field to one value per event; events with no trigger
    #    electron (0-length list -> None via firsts) get padded to -9999
    flat_fields = {
        field: ak.fill_none(ak.firsts(jagged_electrons[field]), -9999)
        for field in jagged_electrons.fields
    }
    electrons = ak.zip(flat_fields, depth_limit=1)

    # 5. Sampling Fraction computed from the now-flat fields; -9999 where it doesn't pass
    sf_raw = (electrons["E_PCAL"] + electrons["E_ECOUT"] + electrons["E_ECIN"]) / electrons["p"]
    sf = ak.where(pass_trigger, sf_raw, -9999.0)
    electrons = ak.with_field(electrons, sf, "SF")

    # 6. Attach back
    events = ak.with_field(events, pass_trigger, "pass_trigger")
    events["reconstructed"] = electrons

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

    electrons = events["reconstructed"]   # now fully flat: one record/event, -9999-padded

    p = electrons["p"]
    theta = electrons["theta"]

    pass_dis = ak.to_numpy(events["pass_trigger"] & (p <= E_beam))

    p_arr = ak.to_numpy(p)
    theta_arr = ak.to_numpy(theta)

    with np.errstate(invalid="ignore", divide="ignore"):
        nu_all = calc_nu(p_arr, E_beam)
        Q2_all = calc_Q2(p_arr, E_beam, theta_arr)
        y_all  = calc_y(p_arr, E_beam)
        W2_all = calc_W2(p_arr, E_beam, theta_arr)
        x_all  = calc_xb(Q2_all, E_beam, nu_all)

    nu = np.where(pass_dis, nu_all, -9999.0)
    Q2 = np.where(pass_dis, Q2_all, -9999.0)
    x  = np.where(pass_dis, x_all,  -9999.0)
    y  = np.where(pass_dis, y_all,  -9999.0)

    valid_W_mask = pass_dis & (W2_all > 0)
    with np.errstate(invalid="ignore"):
        W = np.where(valid_W_mask, np.sqrt(np.where(W2_all > 0, W2_all, 0.0)), -9999.0)

    electrons = ak.with_field(electrons, nu, "nu")
    electrons = ak.with_field(electrons, Q2, "Q2")
    electrons = ak.with_field(electrons, x,  "x")
    electrons = ak.with_field(electrons, y,  "y")
    electrons = ak.with_field(electrons, W,  "W")

    events = ak.with_field(events, electrons, "reconstructed")
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
    
    parameters = LoadYaml(os.path.join(flags.config_directory, flags.config))

    events_array = open_data(
        data_paths = [flags.input_file],
        branches_to_open = parameters["BRANCHES_TO_OPEN"],
        data_tree_name = "data",
        open_MC = flags.save_MC,
        MC_branches_to_open = parameters["MC_BRANCHES_TO_OPEN"] if flags.save_MC else None,
        MC_tree_name = "MC",
    )

    # Removing events with no reconstructed particles
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