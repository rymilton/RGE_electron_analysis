#!/bin/bash
data_directory="/work/clas12/skuditha/skims/phys_val/skim/"
output_directory="/work/clas12/rmilton/rge_datasets/phys_val/020150/banks/"
mkdir -p "${output_directory}"

filename="020150_trigger_e.hipo"
output_filename="banks_020150_trigger_e.root"
./hipo2root "${data_directory}" "${filename}" "${output_directory}" "${output_filename}" "0"