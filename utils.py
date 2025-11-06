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
    MC_tree_name = "MC"
):
    event_dictionary = {"reconstructed": ak.Array([])}
    print("Using file", data_path)
    start_time = time.time()
    with uproot.open(data_path + f":{data_tree_name}") as file:
        print("Opening reconstructed data")
        event_dictionary["reconstructed"] = file.arrays(filter_name = branches_to_open)
    if open_MC:
        event_dictionary["MC"] = ak.Array([])
        with uproot.open(data_path + f":{MC_tree_name}") as file:
            print("Opening MC data")
            event_dictionary["MC"] = file.arrays(filter_name = MC_branches_to_open)
    print(f"Took {time.time()-start_time} s to open file!")
    output_array = ak.Array(event_dictionary)
    print(f"Loaded {len(output_array)} events")
    return output_array