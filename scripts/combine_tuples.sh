#!/bin/bash
run_list=(020131 020132 020133 020134 020135 020136 020137 020138 020139 020140 020141 020143 020144 020145 020148 020149 020150 020151 020152 020153 020154 020155 020156 020158 020160 020161 020162 020163 020164 020165 020166 020167 020168 020169 020170 020171 020172 020173 020174 020175 020176)
cd /work/clas12/rmilton/rge_datasets/pass09/
first_run=${run_list[0]}
last_run=${run_list[${#run_list[@]}-1]}
combine_string="hadd -f tuples_${first_run}-${last_run}_pass09.root"

for run_number in "${run_list[@]}"
do 
    hadd -f ./${run_number}/tuples_${run_number}.root ./${run_number}/tuples/*.root
    combine_string="${combine_string} ./${run_number}/tuples_${run_number}.root"
done

echo "${combine_string}"
eval "${combine_string}"
cd /work/clas12/rmilton/RGE_electron_analysis_git/