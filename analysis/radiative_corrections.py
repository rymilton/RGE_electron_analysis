import numpy as np
import pandas as pd
import analysis_options
import argparse


# Given x and Q2 bin centers, outputs the input that is needed for externals
def InputForExternals(x_bin_centers, Q2_bin_centers, output_file=None):
    Eb = 10.5473
    Mp = 0.938272
    x_bin_centers = np.asarray(x_bin_centers)
    Q2_bin_centers = np.asarray(Q2_bin_centers)
    Ep = Eb - Q2_bin_centers / (2.0 * Mp * x_bin_centers)
    theta = 2.0 * np.arcsin(np.sqrt(Q2_bin_centers / (4.0 * Eb * Ep)))

    theta_deg = np.degrees(theta)
    theta_deg[theta_deg < 0] += 360

    lines = ["E\tEp\ttheta\tW\ty\tx\tQ2"]
    for Ep, theta in zip(Ep, theta_deg):
        if Ep < 0 or np.isnan(Ep) or np.isnan(theta):
            continue
        lines.append(
            "{Eb:.3f} {Ep:.3f} {theta:.4f}".format(
                Eb=round(Eb, 3), Ep=round(Ep, 3), theta=round(theta, 4)
            )
        )

    if output_file:
        with open(output_file, "w") as f:
            f.write("\n".join(lines) + "\n")
        print(f"Wrote externals input to {output_file}")
    else:
        for line in lines:
            print(line)


# Calculates the centroid values of each bin
def CalculateCentroids(x_bin_edges, Q2_bin_edges, x_data, Q2_data):
    x_centroids = []
    Q2_centroids = []
    bin_yields = []
    for i, x_edge in enumerate(x_bin_edges):
        if i == len(x_bin_edges) - 1:
            break
        x_bin_low, x_bin_high = x_edge, x_bin_edges[i + 1]
        for j, Q2_edge in enumerate(Q2_bin_edges):
            if j == len(Q2_bin_edges) - 1:
                break
            Q2_bin_low, Q2_bin_high = Q2_edge, Q2_bin_edges[j + 1]
            bin_mask = (
                (x_data < x_bin_high)
                & (x_data > x_bin_low)
                & (Q2_data < Q2_bin_high)
                & (Q2_data > Q2_bin_low)
            )
            x_in_bin = x_data[bin_mask]
            Q2_in_bin = Q2_data[bin_mask]
            bin_yield = len(x_in_bin)
            if bin_yield != len(Q2_in_bin):
                print("wut")
            bin_yields.append(bin_yield)
            if len(x_in_bin) == 0 or len(Q2_in_bin) == 0:
                continue
            x_center = np.mean(x_in_bin)
            x_centroids.append(x_center)
            y_center = np.mean(Q2_in_bin)
            Q2_centroids.append(y_center)
    return x_centroids, Q2_centroids


# Function to open the file that contains the radiative corrections from externals
def OpenCorrections(file):
    df = pd.read_csv(file, sep="\s+")
    return df


# Applies the radiative and Coulomb corrections in each x, Q2 bin
# The corrections are stored in the corrections_df (df from OpenCorrections)
def ApplyCorrections(Q2_bins, x_bins, counts, counts_errors, corrections_df):
    counts_cc = np.copy(counts)
    counts_rc = np.copy(counts)
    counts_cc_rc = np.copy(counts)
    counts_cc_err = np.copy(counts_errors)
    counts_rc_err = np.copy(counts_errors)
    counts_cc_rc_err = np.copy(counts_errors)

    for _, row in corrections_df.iterrows():
        Q2, x, cc, rc = (
            row["Q2"],
            row["x"],
            row["C_cor"],
            row["Sig_Rad"] / row["Sig_Born"],
        )
        Q2_bin_index, x_bin_index = np.digitize(Q2, Q2_bins), np.digitize(x, x_bins)

        if np.isnan(Q2) or np.isnan(x):
            continue
        if Q2_bin_index >= len(Q2_bins) or x_bin_index >= len(x_bins):
            continue
        counts_cc[x_bin_index - 1][Q2_bin_index - 1] *= cc
        counts_rc[x_bin_index - 1][Q2_bin_index - 1] /= rc
        counts_cc_rc[x_bin_index - 1][Q2_bin_index - 1] *= cc
        counts_cc_rc[x_bin_index - 1][Q2_bin_index - 1] /= rc

        counts_cc_err[x_bin_index - 1][Q2_bin_index - 1] *= cc
        counts_rc_err[x_bin_index - 1][Q2_bin_index - 1] /= rc
        counts_cc_rc_err[x_bin_index - 1][Q2_bin_index - 1] *= cc
        counts_cc_rc_err[x_bin_index - 1][Q2_bin_index - 1] /= rc
    return (
        counts_cc,
        counts_rc,
        counts_cc_rc,
        counts_cc_err,
        counts_rc_err,
        counts_cc_rc_err,
    )


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--target_name",
        default="C",
        help="Target you want the externals input for",
        type=str,
    )
    parser.add_argument(
        "--output_file",
        default="externals_input.txt",
        help="File to save externals input to",
        type=str,
    )

    flags = parser.parse_args()

    return flags


if __name__ == "__main__":
    flags = parse_arguments()
    print(f"Getting EXTERNALS input for {flags.target_name}")

    x_bins = analysis_options.x_bins_by_target[flags.target_name]
    x_bin_centers = (x_bins[1:] + x_bins[:-1]) / 2
    Q2_bins = analysis_options.Q2_bins_by_target[flags.target_name]
    Q2_bin_centers = (Q2_bins[1:] + Q2_bins[:-1]) / 2

    x_padded, Q2_padded = np.meshgrid(x_bin_centers, Q2_bin_centers)
    # Taking transpose gives x constant along inner dimension, Q2 varies along inner dimension
    x_padded = x_padded.T
    Q2_padded = Q2_padded.T

    x_bin_centers_repeated = []
    for xbins in x_padded:
        x_bin_centers_repeated.extend(xbins)
    Q2_bin_centers_repeated = []
    for Q2bins in Q2_padded:
        Q2_bin_centers_repeated.extend(Q2bins)

    InputForExternals(
        x_bin_centers_repeated, Q2_bin_centers_repeated, output_file=flags.output_file
    )
