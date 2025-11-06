import yaml
import os
import uproot
import awkward as ak
import time
def LoadYaml(file_name, base_path="../configs"):
    yaml_path = os.path.join(base_path, file_name)
    return yaml.safe_load(open(yaml_path))

def open_data(
    data_path,
    branches_to_open,
    data_tree_name = "data",
    open_MC = False,
    MC_branches_to_open = None,
    MC_tree_name = "MC",
    nmax = None
):
    event_dictionary = {"reconstructed": ak.Array([])}
    print("Using file", data_path)
    start_time = time.time()
    with uproot.open(data_path + f":{data_tree_name}") as file:
        print("Opening reconstructed data")
        if nmax is not None:
            event_dictionary["reconstructed"] = file.arrays(filter_name = branches_to_open, entry_stop=nmax)
        else:
            event_dictionary["reconstructed"] = file.arrays(filter_name = branches_to_open)
    if open_MC:
        event_dictionary["MC"] = ak.Array([])
        with uproot.open(data_path + f":{MC_tree_name}") as file:
            print("Opening MC data")
            if nmax is not None:
                event_dictionary["MC"] = file.arrays(filter_name = MC_branches_to_open, entry_stop=nmax)
            else:
                event_dictionary["MC"] = file.arrays(filter_name = MC_branches_to_open)
    print(f"Took {time.time()-start_time} s to open file!")
    output_array = ak.Array(event_dictionary)
    print(f"Loaded {len(output_array)} events")
    return output_array
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