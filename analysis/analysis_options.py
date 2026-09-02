import numpy as np

x_bins_by_target = {
    "C": np.linspace(0, 1, num=50 + 1),
    "Cu": np.linspace(0, 1, num=50 + 1),
    "Pb": np.linspace(0, 1, num=50 + 1),
    "Al": np.linspace(0, 1, num=50 + 1),
    "Sn": np.linspace(0, 1, num=50 + 1),
    "LD2": np.linspace(0, 1, num=50 + 1),
}
Q2_bins_by_target = {
    "C": np.logspace(np.log10(1), np.log10(11), num=45 + 1, base=10.0),
    "Cu": np.logspace(np.log10(1), np.log10(11), num=45 + 1, base=10.0),
    "Pb": np.logspace(np.log10(1), np.log10(11), num=45 + 1, base=10.0),
    "Al": np.logspace(np.log10(1), np.log10(11), num=45 + 1, base=10.0),
    "Sn": np.logspace(np.log10(1), np.log10(11), num=45 + 1, base=10.0),
    "LD2": np.logspace(np.log10(1), np.log10(11), num=45 + 1, base=10.0),
}
