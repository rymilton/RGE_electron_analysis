#!/bin/bash


cuts_directory="/volatile/clas12/rmilton/rge_datasets/pass1_withfmt/torus-1/C_D2/cuts/"
# Reference run to develop cuts
run_numbers=(
    020030
)

save_MC_info="0"
njobs=32

for run_number in "${run_numbers[@]}"; do
    echo "=== Run ${run_number} ==="
    run_number_arg=$((10#${run_number}))

    data_directory="/cache/clas12/rg-e/production/spring2024/pass1/torus-1/C_D2/dst/recon/${run_number}/"
    base_output_dir="/volatile/clas12/rmilton/rge_datasets/pass1_withfmt/torus-1/C_D2/${run_number}/"
    plot_directory="${base_output_dir}/plots/"
    output_directory="${base_output_dir}/root/"
    mkdir -p "${output_directory}"
    mkdir -p "${plot_directory}"

    filenames=()
    for filepath in "${data_directory}"*.hipo; do
        filenames+=("$(basename "${filepath}")")
    done

    ./hipo2root "${data_directory}" "${output_directory}" "${save_MC_info}" "${njobs}" "${filenames[@]}"

    root_file_directory=${output_directory}
    tuple_output_directory="${base_output_dir}/ntuples/"
    mkdir -p "${tuple_output_directory}"

    root -l -b -q "./tuple_maker.cpp+(\"${root_file_directory}\", \"${tuple_output_directory}\", 0, ${njobs})"

    filenames=()
    for filepath in "${tuple_output_directory}"*.root; do
        filenames+=("${filepath}")
    done
    output_eventbuilder_directory="${base_output_dir}/eventbuilder/"
    mkdir -p "${output_eventbuilder_directory}"

    python eventbuilder_electron_selection.py --input_file ${filenames[@]} --output_directory "${output_eventbuilder_directory}" --num_processes "${njobs}"

    filenames=()
    for filepath in "${output_eventbuilder_directory}"*.root; do
        filenames+=("${filepath}")
    done
    output_candidates_directory="${base_output_dir}/candidates/"
    mkdir -p "${output_candidates_directory}"
    log_file="${base_output_dir}/electron_selection_log.txt"
    echo ${log_file}
    python electron_selection.py --input_file_array ${filenames[@]} --output_directory "${output_candidates_directory}" --num_processes "${njobs}" --save_plots --target_selection --run_number ${run_number_arg} --plots_directory "${plot_directory}" --log_file "${log_file}" --develop_cuts --cut_directory "${cuts_directory}"
done

# Other runs

run_numbers=(
    020026 020027 020029 020031 020032 020033
    020131 020132 020133 020134 020135 020136 020137 020138 020139
    020140 020141 020142 020143 020144 020145 020148 020149
    020150 020151 020152 020153 020154 020155 020156 020157 020158
    020160 020161 020162 020163 020164 020165 020166 020167 020168 020169
    020170 020171 020172 020173 020174 020175 020176
)

save_MC_info="0"
njobs=32

for run_number in "${run_numbers[@]}"; do
    echo "=== Run ${run_number} ==="
    run_number_arg=$((10#${run_number}))

    data_directory="/cache/clas12/rg-e/production/spring2024/pass1/torus-1/C_D2/dst/recon/${run_number}/"
    base_output_dir="/volatile/clas12/rmilton/rge_datasets/pass1_withfmt/torus-1/C_D2/${run_number}/"
    plot_directory="${base_output_dir}/plots/"
    output_directory="${base_output_dir}/root/"
    mkdir -p "${output_directory}"
    mkdir -p "${plot_directory}"

    filenames=()
    for filepath in "${data_directory}"*.hipo; do
        filenames+=("$(basename "${filepath}")")
    done

    ./hipo2root "${data_directory}" "${output_directory}" "${save_MC_info}" "${njobs}" "${filenames[@]}"

    root_file_directory=${output_directory}
    tuple_output_directory="${base_output_dir}/ntuples/"
    mkdir -p "${tuple_output_directory}"

    root -l -b -q "./tuple_maker.cpp+(\"${root_file_directory}\", \"${tuple_output_directory}\", 0, ${njobs})"

    filenames=()
    for filepath in "${tuple_output_directory}"*.root; do
        filenames+=("${filepath}")
    done
    output_eventbuilder_directory="${base_output_dir}/eventbuilder/"
    mkdir -p "${output_eventbuilder_directory}"

    python eventbuilder_electron_selection.py --input_file ${filenames[@]} --output_directory "${output_eventbuilder_directory}" --num_processes "${njobs}"

    filenames=()
    for filepath in "${output_eventbuilder_directory}"*.root; do
        filenames+=("${filepath}")
    done
    output_candidates_directory="${base_output_dir}/candidates/"
    mkdir -p "${output_candidates_directory}"
    log_file="${base_output_dir}/electron_selection_log.txt"
    echo ${log_file}
    python electron_selection.py --input_file_array ${filenames[@]} --output_directory "${output_candidates_directory}" --num_processes "${njobs}" --save_plots --target_selection --run_number ${run_number_arg} --plots_directory "${plot_directory}" --log_file "${log_file}" --cut_directory "${cuts_directory}"
done
