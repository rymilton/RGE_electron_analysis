#!/bin/bash

python electron_selection.py \
--plots_directory /home/rmilton/work_dir/RGE_electron_analysis_git/plots_LD2Csolid_passreco_clasdis/ \
--save_MC \
--save_plots \
--simulation \
--input_file /home/rmilton/work_dir/rge_datasets/job_9586_LD2Csolid_clasdis_deuteron_zh0_3k/eventbuilder_electrons/electrons_eventbuilder_LD2Csolid_clasdis_deuteron_zh0_3000files.root \
--output_directory /home/rmilton/work_dir/rge_datasets/job_9586_LD2Csolid_clasdis_deuteron_zh0_3k/ \
--output_file candidate_electrons_L2DCsolid_clasdis_deuteron_zh0_3000files_passreco.root

python electron_selection.py \
--plots_directory /home/rmilton/work_dir/RGE_electron_analysis_git/plots_LD2Cliquid_passreco_clasdis/ \
--save_MC \
--save_plots \
--simulation \
--input_file /home/rmilton/work_dir/rge_datasets/job_9663_LD2Cliquid_clasdis_deuteron_zh0_3k/eventbuilder_electrons/electrons_eventbuilder_LD2Cliquid_clasdis_deuteron_zh0_3000files.root \
--output_directory /home/rmilton/work_dir/rge_datasets/job_9663_LD2Cliquid_clasdis_deuteron_zh0_3k/ \
--output_file candidate_electrons_L2DCliquid_clasdis_deuteron_zh0_3000files_passreco.root

python electron_selection.py \
--plots_directory /home/rmilton/work_dir/RGE_electron_analysis_git/plots_carbon_passreco_gibuu/ \
--save_MC \
--save_plots \
--simulation \
--input_file /home/rmilton/work_dir/rge_datasets/job_9481_LD2Csolid_gibuu_carbon_3k/eventbuilder_electrons/electrons_eventbuilder_LD2Csolid_gibuu_carbon_3000files.root \
--output_directory /home/rmilton/work_dir/rge_datasets/job_9481_LD2Csolid_gibuu_carbon_3k/ \
--output_file candidate_electrons_LD2Csolid_gibuu_carbon_3000files_passreco.root

python electron_selection.py \
--plots_directory /home/rmilton/work_dir/RGE_electron_analysis_git/plots_deuterium_passreco_gibuu/ \
--save_MC \
--save_plots \
--simulation \
--input_file /home/rmilton/work_dir/rge_datasets/job_9482_LD2Cliquid_gibuu_deuteron_3k/eventbuilder_electrons/electrons_eventbuilder_LD2Cliquid_gibuu_deuteron_3000files.root \
--output_directory /home/rmilton/work_dir/rge_datasets/job_9482_LD2Cliquid_gibuu_deuteron_3k/ \
--output_file candidate_electrons_LD2Cliquid_gibuu_deuteron_3000files_passreco.root