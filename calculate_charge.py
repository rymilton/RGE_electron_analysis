import argparse
import uproot
import glob
import awkward as ak
import numpy as np
import time
import os

def parse_arguments():
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        "--input_file",
        default="/home/rmilton/work_dir/rge_datasets/pass09/tuples_020131-020176_pass09.root",
        help="ROOT file containing tuples",
        type=str,
    )
    
    flags = parser.parse_args()

    return flags


def main():
    flags = parse_arguments()

    print("Opening file to calculate charge:", flags.input_file)
    with uproot.open(f"{flags.input_file}:meta") as meta:
        meta_info = meta.arrays()
    
    run_numbers = np.unique(meta_info["run_number"])

    total_charge = 0
    total_num_events = 0
    total_num_events_skipped = 0
    
    for run in run_numbers:
        run_data = meta_info[meta_info["run_number"]==run]
        events_in_run = np.sort(np.unique(run_data["event_number"]))
        total_num_events += len(events_in_run)
        data_with_charge = run_data[run_data["fcupgated"]>-1]
        fcupgated_data = data_with_charge["fcupgated"]/1000
        
        max_charge = max(fcupgated_data)
        max_index = ak.argmax(fcupgated_data)
        max_charge_event = data_with_charge["event_number"][max_index]
        num_events_after_max = len(events_in_run[events_in_run>max_charge_event])
        
        min_charge = min(fcupgated_data)
        min_index = ak.argmin(fcupgated_data)
        min_charge_event = data_with_charge["event_number"][min_index]
        num_events_before_min = len(events_in_run[events_in_run<min_charge_event])
    
        total_num_events_skipped += (num_events_after_max + num_events_before_min)
        total_charge += (max_charge - min_charge)
        print(f"Run #{run} has {max_charge-min_charge} microC across {len(events_in_run)}, with {(num_events_after_max + num_events_before_min)} events skipped")

    print(f"Total charge being considered is {round(total_charge/1000, 5)} mC across {total_num_events} events")
    print(f"We skipped { round(total_num_events_skipped/total_num_events*100,2)}% events while calculating")
if __name__ == "__main__":
    main()