#!/bin/bash

run_list=(020131 020132 020133 020134 020135 020136 020137 020138 020139 020140 020141 020143 020144 020145 020148 020149 020150 020151 020152 020153 020154 020155 020156 020158 020160 020161 020162 020163 020164 020165 020166 020167 020168 020169 020170 020171 020172 020173 020174 020175 020176)
working_directory=/work/clas12/rmilton/RGE_electron_analysis_git/

for run_number in "${run_list[@]}"
do
    overall_directory=/work/clas12/rmilton/rge_datasets/pass09/
    banks_directory=/work/clas12/rmilton/rge_datasets/pass09/${run_number}/banks/
    tuples_directory=/work/clas12/rmilton/rge_datasets/pass09/${run_number}/tuples/
    eventbuilder_electrons_directory=/work/clas12/rmilton/rge_datasets/pass09/${run_number}/eventbuilder_electrons/
    mkdir -p "$tuples_directory"
    mkdir -p "$eventbuilder_electrons_directory"

    files=( "${banks_directory}"/* )
    for FILE in "${files[@]}"
    do
        echo "Processing file ${FILE}"
        shortened_file_name=$(echo $FILE| rev | cut -d'/' -f -1 | rev) # Removing directory from file name. Now have banks_XXXXX.root
        file_num=${shortened_file_name#*_} # Remove everything up to and including that underscore. Get XXXXX.root
        file_num=${file_num%%.*} # Remove everything from the . onward, now have XXXXX
        echo ${file_num}
    #     # Running tuple maker
        echo Running tuple maker
        echo root -l -b -q \
        "${working_directory}/tuple_maker.cpp+(\"${banks_directory}\", \"banks_${file_num}.root\", \"${tuples_directory}\", \"ntuples_${file_num}.root\", 0)"

        root -l -b -q "${working_directory}/tuple_maker.cpp+(\"${banks_directory}\", \"banks_${file_num}.root\", \"${tuples_directory}\", \"ntuples_${file_num}.root\", 0)"

        # Running electron selection
        echo Running eventbuilder selection
        python eventbuilder_electron_selection.py \
        --input_file "${tuples_directory}/ntuples_${file_num}.root" \
        --output_directory ${eventbuilder_electrons_directory} \
        --output_file electrons_eventbuilder_${file_num}.root 
    done

    hadd -f /work/clas12/rmilton/rge_datasets/pass09/${run_number}/electrons_eventbuilder_${run_number}.root ${eventbuilder_electrons_directory}/electrons_eventbuilder_*.root
done
cd ${overall_directory}
first_run=${run_list[0]}
last_run=${run_list[${#run_list[@]}-1]}
combine_string="hadd -f electrons_eventbuilder_${first_run}-${last_run}_pass09.root"

for run_number in "${run_list[@]}"
do 
    combine_string="${combine_string} ./${run_number}/electrons_eventbuilder_${run_number}.root"
done

echo "${combine_string}"
eval "${combine_string}"
cd ${working_directory}