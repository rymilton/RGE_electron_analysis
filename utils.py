import yaml
import os
import uproot
import awkward as ak
import time
import numpy as np
import pandas as pd
import concurrent.futures as cf


def CSV_to_df(file_path, separation=","):
    df = pd.read_csv(file_path, sep=separation)
    return df


def LoadYaml(file_name, base_path="../configs"):
    yaml_path = os.path.join(base_path, file_name)
    return yaml.safe_load(open(yaml_path))


def _open_one_file(
    data_path,
    branches_to_open,
    data_tree_name,
    open_gen,
    gen_branches_to_open,
    gen_tree_name,
    get_meta_info,
    entry_stop=None,
):
    # Opens one file's reconstructed/gen/meta_info trees.
    result = {}
    if branches_to_open is not None:
        with uproot.open(data_path + f":{data_tree_name}") as file:
            result["reconstructed"] = file.arrays(
                filter_name=branches_to_open, entry_stop=entry_stop, library="ak"
            )
    if open_gen:
        with uproot.open(data_path + f":{gen_tree_name}") as file:
            result["gen"] = file.arrays(
                filter_name=gen_branches_to_open, entry_stop=entry_stop, library="ak"
            )
    if get_meta_info:
        with uproot.open(data_path + f":meta_info") as file:
            result["meta_info"] = file.arrays(
                filter_name=[
                    "total_luminosity",
                    "total_num_events",
                    "luminosity_after_cuts",
                ],
                entry_stop=entry_stop,
                library="ak",
            )
    return result


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
    num_processes=1,
):
    # Appending the data for each file in lists and then converting to an Awkward array at the end
    # This is more efficienct than repeatedly using ak.concatenate since that copies the whole array every time
    # e.g. have file 1 in an awkward array, and every time we run ak.concatenate we copy file 1
    # this concatenate approach doesn't just append
    reconstructed_parts = []
    gen_parts = []
    meta_parts = []
    if log_file is not None:
        with open(log_file, "a") as f:
            f.write(f"Using file(s) {data_paths}\n")

    # Checking if input file is a list of files or just one file
    if isinstance(data_paths, str):
        data_paths = [data_paths]

    start_time = time.time()

    if num_processes > 1 and len(data_paths) > 1:
        # Parallel path: no per-file nmax truncation possible (workers are
        # dispatched up front, before knowing how many events prior files
        # yielded), so each worker reads its whole file and nmax is enforced
        # by slicing the combined result afterward instead.
        print(f"Opening {len(data_paths)} files using {num_processes} processes...")
        if log_file is not None:
            with open(log_file, "a") as f:
                f.write(
                    f"Opening {len(data_paths)} files using {num_processes} processes\n"
                )
        with cf.ProcessPoolExecutor(max_workers=num_processes) as executor:
            futures = {
                executor.submit(
                    _open_one_file,
                    data_path,
                    branches_to_open,
                    data_tree_name,
                    open_gen,
                    gen_branches_to_open,
                    gen_tree_name,
                    get_meta_info,
                ): data_path
                for data_path in data_paths
            }
            for future in cf.as_completed(futures):
                data_path = futures[future]
                result = future.result()
                if "reconstructed" in result:
                    reconstructed_parts.append(result["reconstructed"])
                if "gen" in result:
                    gen_parts.append(result["gen"])
                if "meta_info" in result:
                    meta_parts.append(result["meta_info"])
                print(f"Opened {data_path}")
                if log_file is not None:
                    with open(log_file, "a") as f:
                        f.write(f"Opened {data_path}\n")
    else:
        remaining_events = nmax
        for data_path in data_paths:
            if nmax is not None and remaining_events <= 0:
                break
            print(f"Opening data from {data_path}")
            if log_file is not None:
                with open(log_file, "a") as f:
                    f.write(f"Opening data from {data_path}\n")

            result = _open_one_file(
                data_path,
                branches_to_open,
                data_tree_name,
                open_gen,
                gen_branches_to_open,
                gen_tree_name,
                get_meta_info,
                entry_stop=remaining_events,
            )

            num_events_in_file = 0
            if "reconstructed" in result:
                reconstructed_parts.append(result["reconstructed"])
                num_events_in_file = len(result["reconstructed"])
            if "gen" in result:
                gen_parts.append(result["gen"])
            if "meta_info" in result:
                meta_parts.append(result["meta_info"])

            if nmax is not None:
                remaining_events -= num_events_in_file
                print(f"{nmax - remaining_events}/{nmax} events loaded")

    event_dictionary = {
        "reconstructed": (
            ak.concatenate(reconstructed_parts) if reconstructed_parts else ak.Array([])
        )
    }
    if open_gen:
        event_dictionary["gen"] = (
            ak.concatenate(gen_parts) if gen_parts else ak.Array([])
        )
    if get_meta_info:
        event_dictionary["meta_info"] = (
            ak.concatenate(meta_parts) if meta_parts else ak.Array([])
        )

    # Safety net for the parallel path (a no-op in the serial path, which
    # already truncates per-file via entry_stop above).
    if nmax is not None:
        event_dictionary["reconstructed"] = event_dictionary["reconstructed"][:nmax]
        if open_gen:
            event_dictionary["gen"] = event_dictionary["gen"][:nmax]
        if get_meta_info:
            event_dictionary["meta_info"] = event_dictionary["meta_info"][:nmax]

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
            if field not in events.fields:
                print(f"{field} not in reconstructed particles. Skipping")
                if log_file is not None:
                    with open(log_file, "a") as f:
                        f.write(f"{field} not in reconstructed particles. Skipping\n")
            continue
        reconstructed_dictionary[field] = events["reconstructed"][field]
    for field in events.fields:
        if field.startswith("pass_"):
            print(f"saving {field}")
            if log_file is not None:
                with open(log_file, "a") as f:
                    f.write(f"saving {field}\n")
            reconstructed_dictionary[field] = events[field]
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
