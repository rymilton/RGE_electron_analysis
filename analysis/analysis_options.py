import json
import os

import numpy as np

# Bin edges derived from the data by analysis/derive_xQ2_binning.py. Every target
# shares the same edges so that target-to-target ratios stay comparable bin by bin.
BINNING_FILE = os.path.join(os.path.dirname(__file__), "xQ2_binning.json")

with open(BINNING_FILE) as binning_file:
    binning = json.load(binning_file)

x_bins_by_target = {
    target: np.asarray(binning["x_edges"]) for target in binning["targets"]
}
Q2_bins_by_target = {
    target: np.asarray(binning["Q2_edges"]) for target in binning["targets"]
}
