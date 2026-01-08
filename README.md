# RGE_electron_analysis
Repository for the inclusive analysis for Run Group E for the CLAS12 experiment at Jefferson Lab. There are three main components to this workflow -- electron selection, unfolding, and cross section computation.

## Environment
The code has been tested on the UCR GPU machine and JLab's ifarm. To set up a Python environment to run the code, do the following:


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
- Packages in package_veresions.txt
- [unbinned_unfolding commit 9b5cdc4](https://github.com/rymilton/unbinned_unfolding/tree/9b5cdc43c211bd364733b78284f9e67679492036)
