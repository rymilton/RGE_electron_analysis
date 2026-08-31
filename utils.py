import yaml
import os
import uproot
import awkward as ak
import time
import numpy as np
import pandas as pd


def CSV_to_df(file_path, separation=","):
    df = pd.read_csv(file_path, sep=separation)
    return df


def LoadYaml(file_name, base_path="../configs"):
    yaml_path = os.path.join(base_path, file_name)
    return yaml.safe_load(open(yaml_path))


def open_data(
    data_paths,
    branches_to_open,
    data_tree_name="data",
    open_gen=False,
    gen_branches_to_open=None,
    gen_tree_name="gen",
    nmax=None,
    output_format="awkward",  # Either dictionary or awkward
    log_file=None,
    get_meta_info=False,
):
    event_dictionary = {"reconstructed": ak.Array([])}
    if open_gen:
        event_dictionary["gen"] = ak.Array([])
    if get_meta_info:
        event_dictionary["meta_info"] = ak.Array([])
    if log_file is not None:
        with open(log_file, "a") as f:
            f.write(f"Using file(s) {data_paths}\n")

    # Checking if input file is a list of files or just one file
    if isinstance(data_paths, str):
        data_paths = [data_paths]

    start_time = time.time()
    if nmax is not None:
        remaining_events = nmax
    for data_path in data_paths:
        if nmax is not None:
            if remaining_events <= 0:
                break
        num_events_in_file = 0
        if branches_to_open is not None:
            with uproot.open(data_path + f":{data_tree_name}") as file:
                print(f"Opening reconstructed data from {data_path}")
                if log_file is not None:
                    with open(log_file, "a") as f:
                        f.write(f"Opening reconstructed data from {data_path}\n")
                if nmax is not None:
                    arrays = file.arrays(
                        filter_name=branches_to_open,
                        entry_start=0,
                        entry_stop=remaining_events,
                        library="ak",
                    )
                    event_dictionary["reconstructed"] = ak.concatenate(
                        (event_dictionary["reconstructed"], arrays)
                    )
                    num_events_in_file = len(arrays)
                    print(
                        f"{nmax-(remaining_events - num_events_in_file)}/{nmax} reco events loaded"
                    )
                else:
                    event_dictionary["reconstructed"] = ak.concatenate(
                        (
                            event_dictionary["reconstructed"],
                            file.arrays(filter_name=branches_to_open, library="ak"),
                        )
                    )
        if open_gen:
            with uproot.open(data_path + f":{gen_tree_name}") as file:
                print("Opening gen data")
                if log_file is not None:
                    with open(log_file, "a") as f:
                        f.write(f"Opening gen data from {data_path}\n")
                if nmax is not None:
                    event_dictionary["gen"] = ak.concatenate(
                        (
                            event_dictionary["gen"],
                            file.arrays(
                                filter_name=gen_branches_to_open,
                                entry_start=0,
                                entry_stop=remaining_events,
                                library="ak",
                            ),
                        )
                    )
                    print(
                        f"{nmax-(remaining_events - num_events_in_file)}/{nmax} gen events loaded"
                    )
                else:
                    event_dictionary["gen"] = ak.concatenate(
                        (
                            event_dictionary["gen"],
                            file.arrays(filter_name=gen_branches_to_open, library="ak"),
                        )
                    )
        if get_meta_info:
            with uproot.open(data_path + f":meta_info") as file:
                print("Opening meta info")
                if log_file is not None:
                    with open(log_file, "a") as f:
                        f.write(f"Opening meta data from {data_path}\n")
                if nmax is not None:
                    event_dictionary["meta_info"] = ak.concatenate(
                        (
                            event_dictionary["meta_info"],
                            file.arrays(
                                filter_name=[
                                    "total_luminosity",
                                    "total_num_events",
                                    "luminosity_after_cuts",
                                ],
                                entry_start=0,
                                entry_stop=remaining_events,
                                library="ak",
                            ),
                        )
                    )
                    print(
                        f"{nmax-(remaining_events - num_events_in_file)}/{nmax} meta events loaded"
                    )
                else:
                    event_dictionary["meta_info"] = ak.concatenate(
                        (
                            event_dictionary["meta_info"],
                            file.arrays(
                                filter_name=[
                                    "total_luminosity",
                                    "total_num_events",
                                    "luminosity_after_cuts",
                                ],
                                library="ak",
                            ),
                        )
                    )
        if nmax is not None:
            remaining_events -= num_events_in_file

    print(f"Took {time.time()-start_time} s to open file!")
    if log_file is not None:
        with open(log_file, "a") as f:
            f.write(f"Took {time.time()-start_time} s to open file!\n")
    if output_format == "dictionary":
        return event_dictionary
    output_array = ak.Array(event_dictionary)
    print(f"Loaded {len(output_array)} events")
    if log_file is not None:
        with open(log_file, "a") as f:
            f.write(f"Loaded {len(output_array)} events\n")
    return output_array


def save_output(
    events,
    output_directory,
    output_file,
    branches_to_save,
    save_gen=False,
    gen_branches_to_save=None,
    log_file=None,
):
    reconstructed_dictionary = {}
    print("Saving reconstructed electrons")
    if log_file is not None:
        with open(log_file, "a") as f:
            f.write("Saving reconstructed electrons\n")
    reconstructed_fields = events["reconstructed"].fields
    for field in branches_to_save:
        if field not in reconstructed_fields:
            print(f"{field} not in reconstructed particles. Skipping")
            if log_file is not None:
                with open(log_file, "a") as f:
                    f.write(f"{field} not in reconstructed particles. Skipping\n")
            continue
        reconstructed_dictionary[field] = events["reconstructed"][field]
    if "pass_reco" in events.fields:
        print("saving pass reco")
        if log_file is not None:
            with open(log_file, "a") as f:
                f.write("saving pass reco\n")
        reconstructed_dictionary["pass_reco"] = events["pass_reco"]
    meta = {}
    if "total_luminosity" in events.fields:
        meta["total_luminosity"] = events["total_luminosity"]
    if "total_num_events" in events.fields:
        meta["total_num_events"] = events["total_num_events"]
    if "luminosity_after_cuts" in events.fields:
        meta["luminosity_after_cuts"] = events["luminosity_after_cuts"]
    if save_gen:
        print("Saving gen electrons")
        if log_file is not None:
            with open(log_file, "a") as f:
                f.write("Saving gen electrons\n")
        gen_dictionary = {}
        gen_fields = events["gen"].fields
        for field in gen_branches_to_save:
            if field not in gen_fields:
                print(f"{field} not in gen particles. Skipping")
                if log_file is not None:
                    with open(log_file, "a") as f:
                        f.write(f"{field} not in gen particles. Skipping\n")
                continue
            gen_dictionary[field] = events["gen"][field]

    os.makedirs(output_directory, exist_ok=True)
    full_output_path = os.path.join(output_directory, output_file)

    with uproot.recreate(full_output_path) as file:
        file["reconstructed_electrons"] = reconstructed_dictionary
        if meta:
            file["meta_info"] = meta
        if save_gen:
            file["gen_electrons"] = gen_dictionary
