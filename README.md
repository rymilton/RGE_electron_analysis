# RGE_electron_analysis
Repository for the inclusive analysis for Run Group E for the CLAS12 experiment at Jefferson Lab. There are three main components to this workflow -- electron selection, unfolding, and cross section computation.

## Environment
The code has been tested on the UCR GPU machine and JLab's ifarm. As a prerequisite, ROOT is required. To set up a Python environment to run the code, do the following:


**First step on ifarm only**:
```
source /etc/profile.d/modules.sh
module use /scigroup/cvmfs/hallb/clas12/sw/modulefiles
module load clas12/5.3
```
**Steps for all systems**:
```
python3 -m venv RGE_electron_analysis_venv
source RGE_electron_analysis_venv/bin/activate
pip install -U pip
pip install matplotlib uproot numpy scipy mplhep pandas scikit-learn pyyaml
git clone https://github.com/rymilton/unbinned_unfolding.git
cd unbinned_unfolding && mkdir build && cd build && make -j12
```
You can then access the environment by doing `source RGE_electron_analysis_venv/bin/activate` followed by `source unbinned_unfolding/build/setup.sh`. Note if you are not interested in doing the unfolding procedure, you can skip the steps about cloning and making the unbinned_unfolding repository.

If on ifarm, `module load clas12` should also work but this code was tested explicitly with the `clas12/5.3` module.

The versions of the code and packages used when testing the code are the following:
- Python 3.9.23
- pip 25.3
- ROOT 6.30/04
- Packages in package_veresions.txt
- [unbinned_unfolding commit 9b5cdc4](https://github.com/rymilton/unbinned_unfolding/tree/9b5cdc43c211bd364733b78284f9e67679492036)

## Electron selection
There are multiple steps in the electron selection procedure:
1. Convert the .hipo data files to .root files
2. Extract relevant quantities and store them in new .root files
3. Select electrons identified by the event builder
4. Refine the electron selection

Step 1 (converting .hipo to .root) uses code external to this repository at this time ([link here](https://github.com/sebouh137/clas12-rge-analysis/)). 
### Tuple maker
`tuple_maker.cpp` takes .root files as input and extracts the quantities that are relevant to this analysis. Every particle in each event is stored. The quantities that are stored can be seen in [these lines](https://github.com/rymilton/RGE_electron_analysis/blob/main/tuple_maker.cpp#L85-L98). In addition to these, the charge from the Faraday cup (fcupgated) and run number are saved. If desired, the Monte Carlo information for each particle is also saved. To run the code, do `root -l -b -q "tuple_maker.cpp+(\"${banks_directory}\", \"banks_${file_num}.root\", \"${tuples_directory}\", \"ntuples_${file_num}.root\", 0)"` where `${banks_directory}` and `banks_${file_num}.root` are the directory and file for the .root file after converting from .hipo, respectively. `${tuples_directory}` and `ntuples_${file_num}.root` are the output directory and output file, respectively. The final argument is `0` if you don't want to save Monte Carlo the information and `1` if you want to save it. This option should only be set to `1` if you're working with simulation files. 

### Event builder electrons
`eventbuilder_electron_selection.py` selects the particles that are identified as electrons by the event builder (pid == 11) and saves those to a .root file. This can be run by doing `python eventbuilder_electron_selection.py --input_file "${tuples_directory}/ntuples_${file_num}.root" --output_directory ${eventbuilder_electrons_directory} --output_file electrons_eventbuilder_${file_num}.root`. If you want to save the Monte Carlo information, also use the option `--save_MC`. This option will only work if you saved the Monte Carlo information during the tuple maker step. The branches that will be opened from the tuple maker .root files and the branches that will be saved are found in `configs/config.yaml` in `BRANCHES_TO_OPEN` and `BRANCHES_TO_SAVE`, respectively. Likewise, `MC_BRANCHES_TO_OPEN` and `MC_BRANCHES_TO_SAVE` are for the Monte Carlo branches. What is opened and saved can be modified, but the current settings will work with the electron selection in the next step.

Note: For the Monte Carlo electrons, the electron with the highest momentum is stored if there are multiple electrons in an event.

### Refining electron selection
To improve the electron selection we apply further cuts. These include kinematic cuts (see `ELECTRON_KINEMATIC_CUTS` in `configs/config.yaml`) and fiducial cuts (see `ELECTRON_FIDUCIAL_CUTS` in `configs/config.yaml`), as well as a partial sampling fraction cut, a tightening of the sampling fraction vs. ECAL energy deposition cut, and for actual data, a z vertex cut to select the targets. More details can be found in the RGE common analysis note. 

To run the code, do the following:
```
python electron_selection.py \
--plots_directory /home/rmilton/work_dir/RGE_electron_analysis_git/plots_LD2Csolid_passreco_clasdis/ \ # The directory to save the plots if --save_plots is used
--save_MC \ # If using simulation
--save_plots \ # If you want to save the plots
--simulation \ # If using simulation
--input_file input.root \ # The input file after running the event builder
--output_directory ./ \ # Directory to store the output files 
--output_file output.root \# Output file name
--target_selection \ # Enable the z vertex cut to select targets. Should be used if using real data
--nmax None \ # Set to a number if you only to cap the number of events used
--solid_target C # Name of solid target
```
The branches to save in the output file are listed in `ELECTRON_SELECTION_BRANCHES_TO_SAVE` in `configs/config.yaml`. Note that these cuts are applied only at reconstruction level. The cuts do not remove any electrons but are instead saved in the `pass_reco` field in the `reconstructed_electrons` branch in the output file. If the electron passes all of the selection cuts, `pass_reco` will be True, otherwise it'll be False.

