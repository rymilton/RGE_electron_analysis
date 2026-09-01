import argparse
import awkward as ak
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from utils import LoadYaml, open_data, save_output


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input_file",
        nargs="+",
        default=[
            "/home/rmilton/work_dir/rge_datasets/job_9586_LD2Csolid_clasdis_deuteron_zh0_3k/ntuples_LD2Csolid_clasdis_deuteron_zh0_3000files.root"
        ],
        help="ROOT file(s) containing tuples from tuple_maker",
        type=str,
    )
    parser.add_argument(
        "--output_directory",
        default="/home/rmilton/work_dir/rge_datasets/job_9586_LD2Csolid_clasdis_deuteron_zh0_3k/eventbuilder_electrons/",
        help="Directory to store event builder electrons",
        type=str,
    )
    parser.add_argument(
        "--num_processes",
        default=4,
        help="Number of processes to use when processing multiple files in parallel",
        type=int,
    )
    parser.add_argument(
        "--save_gen",
        action="store_true",
        default=False,
        help="Load generator-level (Monte Carlo truth) information from file and save it in output electrons",
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


# Event-scalar fields from tuple_maker.cpp (not per-track) -- reused as-is since
# there's at most one trigger-electron candidate per event.
RECONSTRUCTED_EVENT_LEVEL_FIELDS = ["has_trigger_electron", "Q2", "nu", "x", "y", "W"]

# Same idea for the "gen" tree's event-scalar DIS quantities.
GEN_EVENT_LEVEL_FIELDS = ["gen_Q2", "gen_nu", "gen_x", "gen_y", "gen_W"]


def get_eventbuilder_electrons(events):
    # Pull these out before masking -- they're already flat, unlike every other
    # "reconstructed" field.
    event_level_values = {
        field: events["reconstructed"][field]
        for field in RECONSTRUCTED_EVENT_LEVEL_FIELDS
    }

    # 1. Create a mask at the particle level (avoids pid[:, 0] crashes)
    pid = events["reconstructed"]["pid"]
    status = events["reconstructed"]["status"]

    trigger_electron_mask = (pid == 11) & (status <= -2000) & (status > -4000)

    # 2. Filter particles, keeping the jagged event structure intact
    track_level_fields = [
        field
        for field in events["reconstructed"].fields
        if field not in RECONSTRUCTED_EVENT_LEVEL_FIELDS
    ]
    jagged_electrons = events["reconstructed"][track_level_fields][
        trigger_electron_mask
    ]

    # 3. Count trigger electrons per event to build the flag
    number_of_electrons = ak.sum(trigger_electron_mask, axis=1)
    pass_status = number_of_electrons > 0

    if ak.any(number_of_electrons > 1):
        raise ValueError("More than 1 trigger electron found in some events")

    # 4. Flatten to one value per event; no trigger electron -> padded to -9999
    flat_fields = {
        field: ak.fill_none(ak.firsts(jagged_electrons[field]), -9999)
        for field in jagged_electrons.fields
    }
    electrons = ak.zip(flat_fields, depth_limit=1)

    # 5. Sampling Fraction computed from the now-flat fields; -9999 where it doesn't pass
    sf_raw = (
        electrons["E_PCAL"] + electrons["E_ECOUT"] + electrons["E_ECIN"]
    ) / electrons["p"]
    sf = ak.where(pass_status, sf_raw, -9999.0)
    electrons = ak.with_field(electrons, sf, "SF")

    # 6. Reattach the fields pulled out above
    for field, values in event_level_values.items():
        electrons = ak.with_field(electrons, values, field)

    # 7. Attach back
    events = ak.with_field(events, pass_status, "pass_status")
    events["reconstructed"] = electrons

    return events


def get_gen_electrons(events):
    # Same as get_eventbuilder_electrons() above -- pull flat fields out first.
    event_level_values = {
        field: events["gen"][field] for field in GEN_EVENT_LEVEL_FIELDS
    }
    particle_level_fields = [
        field for field in events["gen"].fields if field not in GEN_EVENT_LEVEL_FIELDS
    ]

    events["gen"] = events["gen"][particle_level_fields][events["gen"]["gen_pid"] == 11]

    # Keep the first -- assumed highest-pz (scattered) electron; secondaries are < .5 GeV
    events["gen"] = events["gen"][:, 0]

    for field, values in event_level_values.items():
        events["gen"] = ak.with_field(events["gen"], values, field)

    return events


def derive_output_file_name(input_file):
    stem = os.path.basename(input_file)
    if stem.endswith(".root"):
        stem = stem[: -len(".root")]
    if stem.startswith("ntuples_"):
        stem = stem[len("ntuples_") :]
    return f"electrons_eventbuilder_{stem}.root"


def single_file_eventbuilder_selection(input_file, flags, parameters):
    events_array = open_data(
        data_paths=[input_file],
        branches_to_open=parameters["BRANCHES_TO_OPEN"],
        data_tree_name="reconstructed",
        open_gen=flags.save_gen,
        gen_branches_to_open=(
            parameters["GEN_BRANCHES_TO_OPEN"] if flags.save_gen else None
        ),
        gen_tree_name="gen",
    )

    events_array = get_eventbuilder_electrons(events_array)

    if flags.save_gen:
        events_array = get_gen_electrons(events_array)

    save_output(
        events_array,
        flags.output_directory,
        derive_output_file_name(input_file),
        parameters["BRANCHES_TO_SAVE"],
        flags.save_gen,
        parameters["GEN_BRANCHES_TO_SAVE"] if flags.save_gen else None,
    )


def main():
    flags = parse_arguments()

    parameters = LoadYaml(flags.config, base_path=flags.config_directory)

    njobs = max(1, min(flags.num_processes, len(flags.input_file)))

    if njobs == 1:
        for input_file in flags.input_file:
            try:
                single_file_eventbuilder_selection(input_file, flags, parameters)
                print(f"Done: {input_file}")
            except Exception as e:
                print(f"FAILED: {input_file}: {e}")
    else:
        with ProcessPoolExecutor(max_workers=njobs) as executor:
            futures = {
                executor.submit(
                    single_file_eventbuilder_selection, input_file, flags, parameters
                ): input_file
                for input_file in flags.input_file
            }
            for future in as_completed(futures):
                input_file = futures[future]
                try:
                    future.result()
                    print(f"Done: {input_file}")
                except Exception as e:
                    print(f"FAILED: {input_file}: {e}")


if __name__ == "__main__":
    main()
