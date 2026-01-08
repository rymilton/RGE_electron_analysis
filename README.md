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

Step 1 (converting .hipo to .root) uses code external to this repository at this time ([link here](https://github.com/sebouh137/clas12-rge-analysis/)). Example scripts to run the below steps are found in `scripts`.
### Tuple maker
`tuple_maker.cpp` takes .root files as input and extracts the quantities that are relevant to this analysis. Every particle in each event is stored. The quantities that are stored can be seen in [these lines](https://github.com/rymilton/RGE_electron_analysis/blob/main/tuple_maker.cpp#L85-L98). In addition to these, the charge from the Faraday cup (fcupgated) and run number are saved. If desired, the Monte Carlo information for each particle is also saved. To run the code, do `root -l -b -q "tuple_maker.cpp+(\"${banks_directory}\", \"banks_${file_num}.root\", \"${tuples_directory}\", \"ntuples_${file_num}.root\", 0)"` where `${banks_directory}` and `banks_${file_num}.root` are the directory and file for the .root file after converting from .hipo, respectively. `${tuples_directory}` and `ntuples_${file_num}.root` are the output directory and output file, respectively. The final argument is `0` if you don't want to save Monte Carlo the information and `1` if you want to save it. This option should only be set to `1` if you're working with simulation files. 

### Event builder electrons
`eventbuilder_electron_selection.py` selects the particles that are identified as electrons by the event builder (pid == 11) and saves those to a .root file. This can be run by doing `python eventbuilder_electron_selection.py --input_file "${tuples_directory}/ntuples_${file_num}.root" --output_directory ${eventbuilder_electrons_directory} --output_file electrons_eventbuilder_${file_num}.root`. If you want to save the Monte Carlo information, also use the option `--save_MC`. This option will only work if you saved the Monte Carlo information during the tuple maker step. The branches that will be opened from the tuple maker .root files and the branches that will be saved are found in `configs/config.yaml` in `BRANCHES_TO_OPEN` and `BRANCHES_TO_SAVE`, respectively. Likewise, `MC_BRANCHES_TO_OPEN` and `MC_BRANCHES_TO_SAVE` are for the Monte Carlo branches. What is opened and saved can be modified, but the current settings will work with the electron selection in the next step.

Note: For the Monte Carlo electrons, the electron with the highest momentum is stored if there are multiple electrons in an event.

### Refining electron selection
To improve the electron selection we apply further cuts. These include kinematic cuts (see `ELECTRON_KINEMATIC_CUTS` in `configs/config.yaml`) and fiducial cuts (see `ELECTRON_FIDUCIAL_CUTS` in `configs/config.yaml`), as well as a partial sampling fraction cut, a tightening of the sampling fraction vs. ECAL energy deposition cut, and for actual data, a z vertex cut to select the targets. More details can be found in the RGE common analysis note. Before running the electron selection, you should `hadd` the event builder electron files if your sample is not very large to ensure good fits.

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


## Unfolding

The unfolding in this analysis is unbinned and uses multiple observables at once. The unfolding variables can be set in `./analysis/unfolding_config.yaml`. This file also contains the simulation and data file paths after running the electron selection. There are two main unfolding scripts -- one for a closure test and one for unfolding the RGE data. Instructions for both are below.

### Closure test
To make sure the unfolding works as expected, we perform a closure test. We use two simulation datasets and treat one as simulation data and the other as pseudodata. After unfolding, we should get the truth of the pseudodata. Currently, the code only assumes GiBUU and clasdis files exist (see [here](https://github.com/rymilton/RGE_electron_analysis/blob/main/analysis/unfolding/closure_test.py#L144-L155)). It also assumes the simulation is split into two files -- one for solid target scattering and one for liquid target scattering. To run the code, do the following:
```
python closure_test.py \
--roounfold_path /home/rmilton/work_dir/unbinned_unfolding/build/RooUnfold/ \ # Path to RooUnfold from unbinned_unfolding repository
--load_omnifold_model \ # Enable loading a previously trained omnifold model
--model_path \ # Path to the trained model. This should include the whole path EXCEPT for the _iteration.pkl portion. e.g. clasdis_gibuu_closure instead of clasdis_gibuu_closure_iteration_3.pkl
--num_iterations 4 \ # Number of iterations for the unfolding
--num_events 100000 \ # Number of events for each data sample (after combining the liquid and solid samples)
--MC_name GiBUU \ # Sample to use as simulation
--pseudodata_name clasdis \ # Sample to use as pseudodata
--config ../unfolding_config.yaml \ # Path to config file
--train_test_split \ # Enable if you want to split into train/test for unfolding
--test_fraction 0.2 \ # If train_test_split is enabled, the fraction of data to use for testing
--plot_directory ./ # Path to save plots to
```
### Data unfolding
Once we verify that the closure test is working, we can unfold the actual data. The steps are very similar for this. The code is below:
```
python RGE_unfolding.py \
--roounfold_path /home/rmilton/work_dir/unbinned_unfolding/build/RooUnfold/ \ # Path to RooUnfold from unbinned_unfolding repository
--load_omnifold_model \ # Enable loading a previously trained omnifold model
--model_path \ # Path to the trained model. This should include the whole path EXCEPT for the _iteration.pkl portion. e.g. clasdis_gibuu_closure instead of clasdis_gibuu_closure_iteration_3.pkl
--num_iterations 4 \ # Number of iterations for the unfolding
--num_events 100000 \ # Number of events for each data sample (after combining the liquid and solid samples)
--MC_name GiBUU \ # Sample to use as simulation
--MC2_name clasdis \ # Not used in unfolding. Just added to plots for comparison
--data_file ./RGE_data.root \ # Path to the RGE data file
--config ../unfolding_config.yaml \ # Path to config file
--train_test_split \ # Enable if you want to split into train/test for unfolding
--test_fraction 0.2 \ # If train_test_split is enabled, the fraction of data to use for testing
--plot_directory ./ # Path to save plots to
```

## Cross sections
I'm currently only calculating the cross sections for deuterium and carbon so far. 

### Calculating differential cross section
Once we do the unfolding, we can use the results to calculate the differential cross sections with respect to x and Q2. The notebook to do so is `./analysis/notebooks/RGE_absolute_cross_sections_with_unfolding.ipynb`. This notebook assumes you already have an unfolding model trained. This notebook outputs the cross sections for carbon and deuterium with unfolding and when just using the reconstructed RGE data. These are saved in .csv files. The formula to calculate the differential cross section is below
<img width="558" height="114" alt="image" src="https://github.com/user-attachments/assets/6b3abd95-6dd2-49c7-8e29-89329c8a6adf" />

$\Delta x$ and $\Delta Q^2$ are the bin widths, $N_i$ are the bin counts (which include unfolding weights), $R_i$ and $C_{c,i}$ are the radiative and Coulomb corrections from EXTERNALS, CPB is a conversion factor to get to pb, and $L_{int}$ is the integrated luminosity. To get the integrated luminosity, we need the charge and luminositry info. These can be found at these sites: [luminosity](https://clasweb.jlab.org/clas12online/timelines/rg-e/RGE2024_progress_all_lumi.html) and [charge](https://clasweb.jlab.org/clas12online/timelines/rg-e/RGE2024_progress_all_charge.html). However, if you're using a fraction of the data you need to calculate what fraction of the data you are using. To get your charge, use `calculate_charge.py`. The integrated luminosity is (charge used)/(total charge in runs) * (number of events used in analysis)/(number of total events before reconstruction cuts) * total integrated luminosity.


### Comparing to theory cross sections
To compare to theoretical cross section predictions, we use Yadism. To install yadism, do `pip install 'yadism[mark, box]'`. This code is tested with version [0.13.6](https://pypi.org/project/yadism/0.13.6/). The notebook to do the predictions is `./analysis/notebooks/RGE_yadism_crosssections.ipynb`. The PDF sets used are nCTEQ15HIX_FullNuc_12_6 and nCTEQ15HIX_FullNuc_2_1. We predict $F_2$, $xF_3$, and $F_L$ and use the below formula to get the cross sections

<img width="664" height="131" alt="image" src="https://github.com/user-attachments/assets/190b0257-e018-499b-a37e-162e9e62b1be" />

We then compare the theory cross sections and RGE cross sections in `./analysis/notebooks/RGE_yadism_plots.ipynb`.
