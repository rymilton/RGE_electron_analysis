#!/bin/bash
#SBATCH --job-name=run_020030_full_electron_selection
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=4:00:00
#SBATCH --mem-per-cpu=4000
#SBATCH --output=/volatile/clas12/rmilton/pass1/logs/slurm_%j.out
#SBATCH --error=/volatile/clas12/rmilton/pass1/logs/slurm_%j.err

########################
# Configuration
########################

RUN_NUMBER=020030
HIPO2ROOT_DIR="/work/clas12/rmilton/clas12-rge-analysis_python"
HIPO_DIR="/volatile/clas12/rg-e/production/pass1/mon/recon/"
WORKING_DIR="/work/clas12/rmilton/RGE_electron_analysis_git"
DATA_DIR="/volatile/clas12/rmilton/pass1_data/"

OVERALL_DIR="${DATA_DIR}/${PASS}"

# Electron selection configuration
TARGET_NAME="C"
DATA_NAME="data_${TARGET_NAME}"



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

RUN_DIR="${DATA_DIR}/${RUN_NUMBER}"
BANKS_DIR="${RUN_DIR}/banks"
PLOTS_DIR="${RUN_DIR}/${TARGET_NAME}_${RUN_NUMBER}_plots/"
LOG_FILE="${RUN_DIR}/${TARGET_NAME}_${RUN_NUMBER}_electron_selection_log.txt"

mkdir -p "${RUN_DIR}" "${BANKS_DIR}"

echo "=== Converting run ${RUN_NUMBER} ==="
HIPO_RUN_DIR="${HIPO_DIR}/${RUN_NUMBER}"

files=( $(ls ${HIPO_RUN_DIR}/* | sort) )

for FILE in "${files[@]}"; do
    [[ "$FILE" == *gemc* ]] && continue

    echo "Processing file ${FILE}"

    shortened_file_name=$(basename "$FILE" .hipo)
    file_num="${shortened_file_name}"

    temp_dir="${RUN_DIR}/temp_${file_num}"
    mkdir "${temp_dir}"

    ${HIPO2ROOT_DIR}/bin/hipo2root -w "${temp_dir}" "${FILE}"
    mv "${temp_dir}/banks_"* "${BANKS_DIR}/banks_${file_num}.root"
    rm -rf "${temp_dir}"
done

########################
# Tuple making and eventbuilder electron selection
########################

cd "${WORKING_DIR}"

ELECTRONS_DIR="${RUN_DIR}/eventbuilder_electrons"
TUPLES_DIR="${RUN_DIR}/tuples"

mkdir -p "${ELECTRONS_DIR}" "${TUPLES_DIR}"

echo "=== Processing run ${RUN_NUMBER} ==="

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

# Hadd for this run
hadd -f \
  "${RUN_DIR}/electrons_eventbuilder_${RUN_NUMBER}.root" \
  "${ELECTRONS_DIR}/electrons_eventbuilder_"*.root

########################
# Final electron selection
########################

mkdir -p "${PLOTS_DIR}"

python electron_selection.py \
    --plots_directory "${PLOTS_DIR}" \
    --save_plots \
    --input_file "${RUN_DIR}/electrons_eventbuilder_${RUN_NUMBER}.root" \
    --output_directory "${RUN_DIR}" \
    --output_file "candidate_electrons_${RUN_NUMBER}_${PASS}.root" \
    --target_selection \
    --solid_target "${TARGET_NAME}" \
    --log_file "${LOG_FILE}" \
    --data_name "${DATA_NAME}"

echo "=== Pipeline complete for run ${RUN_NUMBER} ==="
