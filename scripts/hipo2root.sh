#!/bin/bash
data_directory="/cache/clas12/rg-e/production/spring2024/pass1/torus-1/C_D2/dst/recon/020030/"
output_directory="/volatile/clas12/rmilton/rge_datasets/testing/"
save_MC_info="0"
njobs=16

mkdir -p "${output_directory}"

# Collect just the basenames (hipo2root_parallel prepends data_directory itself)
filenames=()
count=0
for filepath in "${data_directory}"*.hipo; do
    filenames+=("$(basename "${filepath}")")
    count=$((count + 1))
done


./hipo2root "${data_directory}" "${output_directory}" "${save_MC_info}" "${njobs}" "${filenames[@]}"