import json
import os

import numpy as np

# Default bin edges, derived from the data by analysis/derive_xQ2_binning.py.
# Every target shares the same edges so that target-to-target ratios stay
# comparable bin by bin.
BINNING_FILE = os.path.join(os.path.dirname(__file__), "xQ2_binning.json")


def get_x_Q2_binning(binning_file=BINNING_FILE):
    """Returns the (x, Q2) bin edges from a binning file written by
    derive_xQ2_binning.py."""
    with open(binning_file) as opened_file:
        binning = json.load(opened_file)
    return np.asarray(binning["x_edges"]), np.asarray(binning["Q2_edges"])
