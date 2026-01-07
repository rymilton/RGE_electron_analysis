import ROOT
import numpy as np
import yaml
import matplotlib.pyplot as plt
from analysis_dataloader import AnalysisDataloader

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
    sim_vals, data_vals, unfolding_weights,
    sim_label, data_label, unfolded_label,
    bins, binning_range, xlabel, title, outfile
):
    counts_sim, edges = np.histogram(sim_vals, bins=bins, range=binning_range)
    counts_data, _ = np.histogram(data_vals, bins=bins, range=binning_range)
    counts_unf, _ = np.histogram(
        sim_vals, bins=bins, range=binning_range, weights=unfolding_weights
    )

    centers = 0.5 * (edges[1:] + edges[:-1])
    width = np.diff(edges)


    fig, axs = plt.subplots(2, 1, figsize=(10, 10), sharex=True, gridspec_kw={
        'height_ratios': [2, 1],   # top panel bigger
        'hspace': 0.05             # reduce space between panels
    })
    axs = axs.flatten()

    axs[0].hist(
        sim_vals,
        bins=bins,
        range=binning_range,
        density=True,
        label=sim_label,
        histtype="step",
        color="#2ca02c",
        linewidth=2,
    )

    axs[0].hist(
        data_vals,
        bins=bins,
        range=binning_range,
        density=True,
        label=data_label,
        histtype="step",
        color="#ff7f0e",
        linewidth=2,
    )

    norm = lambda value, sum_variable: value / (np.sum(sum_variable) * width)

    unfolded_errors = np.sqrt(counts_unf)
    normalized_unfolded_counts = norm(counts_unf, counts_unf)
    normalized_unfolded_errors = norm(unfolded_errors, counts_unf)
    axs[0].errorbar(
        centers,
        normalized_unfolded_counts,
        yerr=normalized_unfolded_errors,
        fmt="o",
        label=unfolded_label,
        color="#1f77b4",
        markersize=7,
    )

    axs[0].set_ylabel("Normalized Entries")
    axs[0].legend(loc="upper right")

    normalized_sim_counts = norm(counts_sim, counts_sim)
    normalized_sim_errors = norm(np.sqrt(counts_sim), counts_sim)
    normalized_data_counts = norm(counts_data, counts_data)
    normalized_data_errors = norm(np.sqrt(counts_data), counts_data)

    sim_truth_ratio, sim_truth_ratio_err = DivideWithErrors(
        normalized_sim_counts,
        normalized_sim_errors,
        normalized_unfolded_counts,
        normalized_unfolded_errors,
    )
    data_truth_ratio, data_truth_ratio_err = DivideWithErrors(
        normalized_data_counts,
        normalized_data_errors,
        normalized_unfolded_counts,
        normalized_unfolded_errors,
    )

    axs[1].errorbar(
        centers,
        sim_truth_ratio,
        yerr=sim_truth_ratio_err,
        fmt="o",
        color="#2ca02c",
        markersize=7,
    )

    axs[1].errorbar(
        centers,
        data_truth_ratio,
        yerr=data_truth_ratio_err,
        fmt="o",
        color="#ff7f0e",
        markersize=7,
    )
    axs[1].axhline(1.0, color="red", linestyle="--", linewidth=1.5)
    axs[1].set_ylabel("Counts/Unfolded")
    axs[1].set_xlabel(xlabel)
    axs[1].set_ylim(0.5, 1.5)

    plt.tight_layout()

    plt.suptitle(title, y=.95)
    plt.savefig(outfile)
    plt.close()

def unfolding_procedure(
    flags,
    simulation_dataloader,
    data_dataloader,
    variables_to_unfold,
):

    if not flags.load_omnifold_model:
        print("Setting up training data dictionaries")
        simulation_training = simulation_dataloader.get_training_data()
        data_training = data_dataloader.get_training_data()


        sim_MCreco_dict_train, sim_MCgen_dict_train, data_dict_train = {}, {}, {}
        for variable in variables_to_unfold:
            sim_MCreco_dict_train[variable] = np.array(simulation_training[0][variable])
            sim_MCgen_dict_train[variable] = np.array(simulation_training[1][f"MC_{variable}"])
            data_dict_train[variable] = np.array(data_training[0][variable])
        df_MCgen_train = ROOT.RDF.FromNumpy(sim_MCgen_dict_train)
        df_MCreco_train = ROOT.RDF.FromNumpy(sim_MCreco_dict_train)
        df_measured_train = ROOT.RDF.FromNumpy(data_dict_train)
        sim_pass_reco_vector_train = np_to_TVector(simulation_training[2])
        data_pass_reco_vector_train = np_to_TVector(data_training[2])

        print("Training omnifold model")
        unbinned_unfolding = ROOT.RooUnfoldOmnifold()
        unbinned_unfolding.SetSaveDirectory("./")
        unbinned_unfolding.SetModelSaveName("clasdis_gibuu_closure")
        unbinned_unfolding.SetMCgenDataFrame(df_MCgen_train)
        unbinned_unfolding.SetMCrecoDataFrame(df_MCreco_train)
        unbinned_unfolding.SetMCPassReco(sim_pass_reco_vector_train)
        unbinned_unfolding.SetMeasuredDataFrame(df_measured_train)
        unbinned_unfolding.SetMeasuredPassReco(data_pass_reco_vector_train)
        unbinned_unfolding.SetNumIterations(flags.num_iterations)
        _ = unbinned_unfolding.UnbinnedOmnifold()

    simulation_testing = simulation_dataloader.get_testing_data()
    data_testing = data_dataloader.get_testing_data()

    sim_MCreco_dict_test, sim_MCgen_dict_test, data_dict_test = {}, {}, {}
    for variable in variables_to_unfold:
        sim_MCreco_dict_test[variable] = np.array(simulation_testing[0][variable])
        sim_MCgen_dict_test[variable] = np.array(simulation_testing[1][f"MC_{variable}"])
        data_dict_test[variable] = np.array(data_testing[0][variable])
    df_MCgen_test = ROOT.RDF.FromNumpy(sim_MCgen_dict_test)
    df_MCreco_test = ROOT.RDF.FromNumpy(sim_MCreco_dict_test)
    df_measured_test = ROOT.RDF.FromNumpy(data_dict_test)
    sim_pass_reco_vector_test = np_to_TVector(simulation_testing[2])
    data_pass_reco_vector_test = np_to_TVector(data_testing[2])

    model_name = "clasdis_gibuu_closure" if flags.model_path is None else flags.model_path
    
    unbinned_unfolding = ROOT.RooUnfoldOmnifold()
    unbinned_unfolding.SetTestMCgenDataFrame(df_MCgen_test)
    unbinned_unfolding.SetTestMCrecoDataFrame(df_MCreco_test)
    unbinned_unfolding.SetTestMCPassReco(sim_pass_reco_vector_test)
    unbinned_unfolding.SetLoadModelPath(f"{model_name}_iteration_0.pkl")
    test_unbinned_results = unbinned_unfolding.TestUnbinnedOmnifold()
    step1_weights = TVector_to_np(ROOT.std.get[0](test_unbinned_results))
    unbinned_unfolding.SetTestMCgenDataFrame(df_MCgen_test)
    unbinned_unfolding.SetTestMCrecoDataFrame(df_MCreco_test)
    unbinned_unfolding.SetTestMCPassReco(sim_pass_reco_vector_test)
    unbinned_unfolding.SetLoadModelPath(f"{model_name}_iteration_{flags.num_iterations-1}.pkl")
    test_unbinned_results = unbinned_unfolding.TestUnbinnedOmnifold()
    step2_weights = TVector_to_np(ROOT.std.get[1](test_unbinned_results))

    return step1_weights, step2_weights