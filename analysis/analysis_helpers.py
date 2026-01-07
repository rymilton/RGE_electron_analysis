import ROOT
import numpy as np
import yaml
import matplotlib.pyplot as plt

def DivideWithErrors(numerator, numerator_error, dividend, dividend_error):
    quotient = numerator/dividend
    error = quotient * np.sqrt((numerator_error/numerator)**2 + (dividend_error/dividend)**2)
    return quotient, error
def np_to_TVector(array):
    vector = ROOT.TVector(len(array))
    for i, entry in enumerate(array):
        vector[i] = entry
    return vector
def TVector_to_np(vector):
    out_array = []
    for i in range(vector.GetNoElements()):
        out_array.append(vector[i])
    return np.array(out_array)

def plot_unfolded(
    sim_vals, data_vals, unfolded_vals, weights,
    bins, rng, xlabel, title, outfile
):
    counts_sim, edges = np.histogram(sim_vals, bins=bins, range=rng)
    counts_data, _ = np.histogram(data_vals, bins=bins, range=rng)
    counts_unf, _ = np.histogram(
        sim_vals, bins=bins, range=rng, weights=weights
    )

    centers = 0.5 * (edges[1:] + edges[:-1])
    width = np.diff(edges)

    norm = lambda c: c / (np.sum(c) * width)

    fig, (ax, rax) = plt.subplots(
        2, 1, sharex=True,
        gridspec_kw={"height_ratios": [2, 1]}
    )

    ax.step(centers, norm(counts_sim), where="mid", label="MC")
    ax.step(centers, norm(counts_data), where="mid", label="Data")
    ax.errorbar(centers, norm(counts_unf), yerr=np.sqrt(counts_unf), fmt="o")

    ratio, ratio_err = DivideWithErrors(
        norm(counts_sim), np.sqrt(counts_sim),
        norm(counts_unf), np.sqrt(counts_unf)
    )

    rax.errorbar(centers, ratio, yerr=ratio_err, fmt="o")
    rax.axhline(1, ls="--")

    ax.set_ylabel("Normalized")
    rax.set_xlabel(xlabel)
    rax.set_ylabel("MC / Unfolded")
    ax.legend()

    plt.suptitle(title)
    plt.savefig(outfile)
    plt.close()
