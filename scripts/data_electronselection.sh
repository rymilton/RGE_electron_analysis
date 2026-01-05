#!/bin/bash

python electron_selection.py \
--plots_directory /home/rmilton/work_dir/RGE_electron_analysis_git/plots_pass09/ \
--save_plots \
--input_file /home/rmilton/work_dir/rge_datasets/pass09/electrons_eventbuilder_020131-020176_pass09.root \
--output_directory /home/rmilton/work_dir/rge_datasets/pass09/ \
--output_file candidate_electrons_020131-020176_pass09.root \
--target_selection \
--solid_target C