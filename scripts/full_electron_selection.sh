#!/bin/bash

########################
# Configuration
########################

RUN_LIST=(
020185
020186
020187
020191
020192
020193
020194
020195
020196
020197
020198
020199
020200
020201
020202
020203
020205
020207
020208
020209
020214
020215
020216
020217
020219
020220
020221
020222
020223
020224
020225
020226
020227
020228
020229
020230
020231
)

HIPO2ROOT_DIR="/work/clas12/rmilton/clas12-rge-analysis_python"
HIPO_DIR="/volatile/clas12/rg-e/production/pass0.10/mon/recon/"
PASS="pass10"
WORKING_DIR="/work/clas12/rmilton/RGE_electron_analysis_git"
DATA_DIR="/work/clas12/rmilton/rge_datasets"

OVERALL_DIR="${DATA_DIR}/${PASS}"


# Electron selection configuration
TARGET_NAME="Cu"
DATA_NAME="data_${TARGET_NAME}"

PLOTS_DIR="${DATA_DIR}/${PASS}/${TARGET_NAME}_plots/"
LOG_FILE="${DATA_DIR}/${PASS}/${TARGET_NAME}_electron_selection_log.txt"

########################
# Helper functions
########################

short_file_num() {
    local file="$1"
    local base
    base=$(basename "$file")
    echo "${base#*_}" | sed 's/\.root$//'
}

########################
# Hipo to Root conversion
########################

for run_number in "${RUN_LIST[@]}"; do
    RUN_DIR="${DATA_DIR}/${PASS}/${run_number}"
    BANKS_DIR="${RUN_DIR}/banks"

    mkdir -p "${RUN_DIR}" "${BANKS_DIR}" 
    echo "=== Converting run ${run_number} ==="
    HIPO_RUN_DIR=${HIPO_DIR}/${run_number}

    files=( $(ls ${HIPO_RUN_DIR}/* | sort) )
    echo ${files}
    for FILE in "${files[@]}"
    do
        if [[ "$FILE" == *gemc* ]]; then
            continue
        fi
        echo "Processing file ${FILE}"
        shortened_file_name=$(echo $FILE| rev | cut -d'/' -f -1 | rev) # Removing directory from file name
        shortened_file_name=$(echo $shortened_file_name| rev | cut -d'.' -f -2 | rev) # String is now file_num.hipo
        file_num=$(echo $shortened_file_name | cut -d'.' -f 1) # Extracting file_num
        echo ${file_num}
        temp_dir=./temp_${file_num}
        mkdir ${temp_dir}
        ${HIPO2ROOT_DIR}/bin/hipo2root -w ${temp_dir} ${FILE}
        mv ${temp_dir}/banks_* ${BANKS_DIR}/banks_${file_num}.root
        rm -rf ${temp_dir}
    done
done

cd ${WORKING_DIR}
# ########################
# # Tuple making and eventbuilder electron selection
# ########################

for run_number in "${RUN_LIST[@]}"; do
    RUN_DIR="${DATA_DIR}/${PASS}/${run_number}"
    ELECTRONS_DIR="${RUN_DIR}/eventbuilder_electrons"
    TUPLES_DIR="${RUN_DIR}/tuples"
    BANKS_DIR="${RUN_DIR}/banks"
    mkdir -p "${RUN_DIR}" "${ELECTRONS_DIR}" "${TUPLES_DIR}"
    echo "=== Processing run ${run_number} ==="

    for FILE in "${BANKS_DIR}"/*.root; do
        [[ -e "$FILE" ]] || continue

        file_num=$(short_file_num "$FILE")
        echo "Processing file ${file_num}"

        # Tuple maker
        root -l -b -q \
        "${WORKING_DIR}/tuple_maker.cpp+(\"${BANKS_DIR}\", \"banks_${file_num}.root\", \"${TUPLES_DIR}\", \"ntuples_${file_num}.root\", 0)"

        # Eventbuilder electron selection
        python eventbuilder_electron_selection.py \
            --input_file "${TUPLES_DIR}/ntuples_${file_num}.root" \
            --output_directory "${ELECTRONS_DIR}" \
            --output_file "electrons_eventbuilder_${file_num}.root"
    done

    # Hadd per run
    hadd -f \
        "${RUN_DIR}/electrons_eventbuilder_${run_number}.root" \
        "${ELECTRONS_DIR}/electrons_eventbuilder_"*.root
done

########################
# Combine all runs
########################

cd "${OVERALL_DIR}"

first_run="${RUN_LIST[0]}"
last_run="${RUN_LIST[-1]}"

COMBINED_EB_FILE="electrons_eventbuilder_${first_run}-${last_run}_${PASS}.root"

combine_cmd="hadd -f ${COMBINED_EB_FILE}"
for run_number in "${RUN_LIST[@]}"; do
    combine_cmd+=" ./${run_number}/electrons_eventbuilder_${run_number}.root"
done

echo "Running: ${combine_cmd}"
eval "${combine_cmd}"

########################
# Final electron selection
########################

mkdir -p "${PLOTS_DIR}"
cd "${WORKING_DIR}"
python electron_selection.py \
    --plots_directory "${PLOTS_DIR}" \
    --save_plots \
    --input_file "${OVERALL_DIR}/${COMBINED_EB_FILE}" \
    --output_directory "${OVERALL_DIR}" \
    --output_file "candidate_electrons_${first_run}-${last_run}_${PASS}.root" \
    --target_selection \
    --solid_target "${TARGET_NAME}" \
    --log_file "${LOG_FILE}" \
    --data_name "${DATA_NAME}"

echo "=== Pipeline complete ==="
