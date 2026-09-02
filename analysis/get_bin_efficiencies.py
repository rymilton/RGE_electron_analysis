import sys
import os

REPO_TOP_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO_TOP_DIR)
ANALYSIS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ANALYSIS_DIR)

import analysis_options
import concurrent.futures as cf
import awkward as ak
import argparse
from utils import LoadYaml, open_data, save_output, CSV_to_df
import numpy as np
import matplotlib.pyplot as plt
import mplhep as hep

hep.style.use(hep.style.CMS)
def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--output_directory",
        default="/volatile/clas12/rmilton/rge_datasets/pass1/torus-1/C_D2/",
        help="Directory to candidate electrons",
        type=str,
    )
    parser.add_argument(
        "--output_file",
        default="efficiency.csv",
        help="Name of output csv file with efficiencies",
        type=str,
    )
    parser.add_argument(
        "--save_MC",
        action="store_true",
        default=False,
        help="Load Monte Carlo information from file and save it in output electrons",
    )
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Basic config file containing general options",
        type=str,
    )
    parser.add_argument(
        "--config_directory",
        default="./configs/",
        help="Directory containing the config files",
        type=str,
    )
    parser.add_argument(
        "--plots_directory",
        default="./plots/",
        help="Directory to store plots",
        type=str,
    )
    parser.add_argument(
        "--save_plots",
        action="store_true",
        default=False,
        help="Save the plots that are generated during the analysis",
    )
    parser.add_argument(
        "--solid_target",
        default="C",
        help="Name of solid target",
        type=str,
    )
    flags = parser.parse_args()

    return flags

def _open_one(path, branches_to_open, MC_branches_to_open, open_MC, nmax, log_file):
    return open_data(
        data_paths=[path],
        branches_to_open=branches_to_open,
        data_tree_name="reconstructed_electrons",
        open_MC=open_MC,
        MC_branches_to_open=MC_branches_to_open,
        MC_tree_name="MC_electrons",
        nmax=nmax,
        log_file=log_file,
    )

def open_and_combine_files(input_files, flags, parameters):
    """Opens all input files in parallel, tags each event with which file it
    came from, and concatenates into one combined array. The tag lets us
    split the array back apart after cuts/plots are computed once on the
    combined statistics."""
    results_by_index = {}
    with cf.ProcessPoolExecutor() as executor:
        futures = {
            executor.submit(
                _open_one,
                path,
                branches_to_open = ["x", "Q2", "pass_reco", "pass_trigger"],
                MC_branches_to_open = ["MC_x", "MC_Q2", "MC_y", "MC_W"],
                open_MC = True,
                nmax = None,
                log_file=None
            ): i
            for i, path in enumerate(input_files)
        }
        for future in cf.as_completed(futures):
            i = futures[future]
            try:
                results_by_index[i] = future.result()
            except Exception:
                print(f"open_data failed on {input_files[i]}")
                raise

    tagged_results = []
    for i in range(len(input_files)):
        arr = results_by_index[i]
        arr = ak.with_field(arr, np.full(len(arr), i, dtype=np.int64), "source_file_index")
        tagged_results.append(arr)

    return ak.concatenate(tagged_results)

def main():
    flags = parse_arguments()
    parameters = LoadYaml(os.path.join(flags.config_directory, flags.config))

    simulation_files = ["/home/rmilton/work_dir/rge_datasets/clasdis_carbon/candidate_electrons/electrons_eventbuilder_LD2Cliquid_clasdis_deuteron_zh0_3000files.root", "/home/rmilton/work_dir/rge_datasets/clasdis_carbon/candidate_electrons/electrons_eventbuilder_LD2Cliquid_clasdis_deuteron_zh0_3000files.root"]

    events_array = open_and_combine_files(simulation_files, flags, parameters)
    # fiducial_mask = (events_array["MC"]["MC_y"] < 0.85) & (events_array["MC"]["MC_Q2"] > 1) & (events_array["MC"]["MC_W"] > 2)
    # events_array = events_array[fiducial_mask]
    # ---------------------------
    # Define binning
    # ---------------------------
    Q2_binning = analysis_options.Q2_bins_by_target[flags.solid_target]
    x_binning = analysis_options.x_bins_by_target[flags.solid_target]
    x_centers = 0.5 * (x_binning[:-1] + x_binning[1:])
    Q2_centers = 0.5 * (Q2_binning[:-1] + Q2_binning[1:])

    # ---------------------------
    # RECONSTRUCTED (numerator)
    # ---------------------------
    reco = events_array["reconstructed"]

    reco_mask = reco["pass_reco"] & reco["pass_trigger"]

    reco_x = reco["x"][reco_mask]
    reco_Q2 = reco["Q2"][reco_mask]

    N_rec, _, _ = np.histogram2d(
        ak.to_numpy(reco_x),
        ak.to_numpy(reco_Q2),
        bins=(x_binning, Q2_binning),
    )

    # ---------------------------
    # MC (denominator)
    # ---------------------------
    mc = events_array["MC"]

    mc_x = mc["MC_x"]
    mc_Q2 = mc["MC_Q2"]

    N_MC, _, _ = np.histogram2d(
        ak.to_numpy(mc_x),
        ak.to_numpy(mc_Q2),
        bins=(x_binning, Q2_binning),
    )

    # ---------------------------
    # Efficiency
    # ---------------------------
    epsilon = np.zeros_like(N_rec, dtype=float)

    mask = N_MC > 0
    epsilon[mask] = N_rec[mask] / N_MC[mask]

    # Optional: handle empty bins safely
    epsilon[~mask] = np.nan
    print(epsilon)

    plt.figure(figsize=(12, 8))

    # Note: histogram2d returns shape [x_bins-1, Q2_bins-1]
    plt.imshow(
        epsilon.T,  # transpose so Q2 is vertical axis
        origin="lower",
        aspect="auto",
        extent=[
            x_binning[0],
            x_binning[-1],
            Q2_binning[0],
            Q2_binning[-1],
        ],
    )

    plt.colorbar(label="Efficiency $(N_{rec} / N_{MC})$")

    plt.xlabel("x")
    plt.ylabel("Q² (GeV²)")
    plt.title("RG-E LD2+C: clasdis")

    plt.tight_layout()
    plt.savefig("./plots/C_D2/efficiencies.png")

    import pandas as pd

    efficiency_error = np.full_like(epsilon, np.nan, dtype=float)
    valid = N_MC > 0
    efficiency_error[valid] = np.sqrt(
        np.clip(epsilon[valid] * (1 - epsilon[valid]), 0, None) / N_MC[valid]
    )

    rows = []
    for i, x_c in enumerate(x_centers):
        for j, q2_c in enumerate(Q2_centers):
            rows.append({
                "x": x_c,
                "Q2": q2_c,
                "efficiency": epsilon[i, j],
                "efficiency_error": efficiency_error[i, j],
                "N_rec": N_rec[i, j],
                "N_MC": N_MC[i, j],
            })
    df_eff = pd.DataFrame(rows)
    out_file = os.path.join(flags.output_directory, flags.output_file)
    df_eff.to_csv(out_file, index=False, na_rep="nan")
    print(f"Saved efficiency map → {out_file}")

    # ---------------------------
    # Efficiency vs. x, one panel per Q2 bin
    # ---------------------------
    n_Q2_bins = len(Q2_centers)
    ncols = int(np.ceil(np.sqrt(n_Q2_bins)))
    nrows = int(np.ceil(n_Q2_bins / ncols))

    fig, axes = plt.subplots(
        nrows, ncols, figsize=(4 * ncols, 3.5 * nrows)
    )
    axes_flat = np.array(axes).reshape(-1)

    for j in range(n_Q2_bins):
        ax = axes_flat[j]
        valid = np.isfinite(epsilon[:, j])
        ax.errorbar(
            x_centers[valid],
            epsilon[:, j][valid],
            yerr=efficiency_error[:, j][valid],
            fmt="o",
            markersize=4,
            capsize=2,
        )
        ax.set_title(f"${Q2_binning[j]:.2f} < Q^2 < {Q2_binning[j+1]:.2f}$", fontsize=10)
        ax.set_ylim(0, 1.05)
        ax.grid(alpha=0.3)
        ax.set_xlabel("x")
        ax.set_ylabel("$N_{rec} / N_{MC}$")

    # Hide any unused panels (when n_Q2_bins doesn't fill the grid exactly)
    for k in range(n_Q2_bins, len(axes_flat)):
        axes_flat[k].axis("off")

    fig.suptitle("RG-E LD2+C: clasdis")
    fig.tight_layout(rect=[0, 0.03, 1, 0.97])

    grid_plot_path = "./plots/C_D2/efficiencies_by_Q2_grid.png"
    fig.savefig(grid_plot_path)
    plt.close(fig)
    print(f"Saved efficiency grid plot → {grid_plot_path}")

if __name__ == "__main__":
    main()
