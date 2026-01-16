import matplotlib.pyplot as plt
import matplotlib.colors as colors
from mpl_toolkits.axes_grid1 import make_axes_locatable
import mplhep as hep 
hep.style.use(hep.style.CMS)
import numpy as np 
import awkward as ak
import pandas as pd
from scipy.optimize import curve_fit

# Converting operator strings to operations on Awkward arrays
array_operator_dict = {
        '>':  (lambda array,value: array > value),
        '<':  (lambda array,value: array < value),
        '>=': (lambda array,value: array >= value),
        '<=': (lambda array,value: array <= value),
        '==': (lambda array,value: array == value),
        '!=': (lambda array,value: array != value),
    }
num_sectors = 6

# Applying kinematic cuts to each electron based on ELECTRON_KINEMATIC_CUTS in the config file
def apply_kinematic_cuts(events, kinematic_cuts, log_file = None, number_of_initial_electrons = None):
    mask = np.ones(len(events), dtype=bool)
    for cut in kinematic_cuts:
        variable_name, operation, cut_value = cut.split()
        if operation not in array_operator_dict:
            raise ValueError(f"Unsupported operation: {operation}")
        
        mask = (mask) & (array_operator_dict[operation](events["reconstructed"][variable_name], float(cut_value)))
    events["pass_reco"] = mask
    print(f"Have {ak.sum(events['pass_reco'])} events after kinematic cuts")
    if log_file is not None:
        with open(log_file, "a") as f:
            f.write(f"Have {ak.sum(events['pass_reco'])} events after kinematic cuts\n")
    if log_file is not None:
        with open(log_file, "a") as f:
            f.write(f"Have {ak.sum(events['pass_reco'])/number_of_initial_electrons} fraction of events passing kinematic cuts\n")
    return events

# Applying fiducial cuts to each electron based on ELECTRON_FIDUCIAL_CUTS in the config file
def apply_fiducial_cuts(events, fiducial_cuts, save_plots = True, plots_directory = None, plot_title = None, log_file = None, number_of_initial_electrons = None):
    

    PCAL_V_cut, PCAL_W_cut = None, None
    PCAL_fiducial_mask = np.ones(len(events), dtype=bool)
    DC_region1_cut, DC_region2_cut, DC_region3_cut = None, None, None
    DC_fiducial_mask = np.ones(len(events), dtype=bool)
    for cut in fiducial_cuts:
        variable_name, operation, cut_value = cut.split()
        if variable_name == "PCAL_V":
            PCAL_fiducial_mask = (PCAL_fiducial_mask) & (array_operator_dict[operation](events["reconstructed"][variable_name], float(cut_value)))
            PCAL_V_cut = float(cut_value)
        elif variable_name == "PCAL_W":
            PCAL_fiducial_mask = (PCAL_fiducial_mask) & (array_operator_dict[operation](events["reconstructed"][variable_name], float(cut_value)))
            PCAL_W_cut = float(cut_value)
        elif variable_name == "DC_region1_edge":
            DC_region1_mask = (array_operator_dict[operation](events["reconstructed"][variable_name], float(cut_value)))
            DC_fiducial_mask = (DC_fiducial_mask) & (DC_region1_mask)
            DC_region1_cut = float(cut_value)
        elif variable_name == "DC_region2_edge":
            DC_region2_mask = (array_operator_dict[operation](events["reconstructed"][variable_name], float(cut_value)))
            DC_fiducial_mask = (DC_fiducial_mask) & (DC_region2_mask)
            DC_region2_cut = float(cut_value)
        elif variable_name == "DC_region3_edge":
            DC_region3_mask = (array_operator_dict[operation](events["reconstructed"][variable_name], float(cut_value)))
            DC_fiducial_mask = (DC_fiducial_mask) & (DC_region3_mask)
            DC_region3_cut = float(cut_value)
    if save_plots:

        # Plotting the PCAL cuts
        low_bin, high_bin, num_bins = (0, 30), (0,.35), (100, 100)
        fig, axs = plt.subplots(1, 2, figsize=(12,6))
        _, _, _, mesh = axs[0].hist2d(
            np.array(events["reconstructed"]["PCAL_V"]),
            np.array(events["reconstructed"]["SF"]),
            bins = num_bins,
            range=(low_bin, high_bin),
            norm=colors.LogNorm(),
        )
        axs[0].set_ylabel("SF")
        axs[0].set_xlabel("PCAL V (cm)")
        if PCAL_V_cut is not None:
            axs[0].vlines(PCAL_V_cut, low_bin[0], low_bin[1], color='red')

        divider = make_axes_locatable(axs[0])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = fig.colorbar(mesh, cax=cax)

        _, _, _, mesh = axs[1].hist2d(
            np.array(events["reconstructed"]["PCAL_W"]),
            np.array(events["reconstructed"]["SF"]),
            bins = num_bins,
            range=(low_bin, high_bin),
            norm=colors.LogNorm(),
        )
        axs[1].set_ylabel("SF")
        axs[1].set_xlabel("PCAL W (cm)")
        if PCAL_W_cut is not None:
            axs[1].vlines(PCAL_W_cut, low_bin[0], low_bin[1], color='red')
        divider = make_axes_locatable(axs[1])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = fig.colorbar(mesh, cax=cax)
        plt.tight_layout()
        if plot_title is not None:
            plt.suptitle(plot_title, y=1.0)
        if plots_directory is not None:
            plt.savefig(plots_directory+"PCAL_W_V.png")

        low_bin, high_bin, num_bins = (0, 30), (0,.35), (100, 100)
        fig, axs = plt.subplots(1, 2, figsize=(12,6))
        _, _, _, mesh = axs[0].hist2d(
            np.array(events["reconstructed"]["PCAL_U"]),
            np.array(events["reconstructed"]["SF"]),
            bins = num_bins,
            range=(low_bin, high_bin),
            norm=colors.LogNorm(),
        )
        axs[0].set_ylabel("SF")
        axs[0].set_xlabel("PCAL U (cm)")
        divider = make_axes_locatable(axs[0])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = fig.colorbar(mesh, cax=cax)

        _, _, _, mesh = axs[1].hist2d(
            np.array(events["reconstructed"]["PCAL_U"][PCAL_fiducial_mask]),
            np.array(events["reconstructed"]["SF"][PCAL_fiducial_mask]),
            bins = num_bins,
            range=(low_bin, high_bin),
            norm=colors.LogNorm(),
        )
        axs[1].set_ylabel("SF")
        axs[1].set_xlabel("PCAL U (cm)")
        divider = make_axes_locatable(axs[1])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = fig.colorbar(mesh, cax=cax)
        plt.tight_layout()
        if plot_title is not None:
            plt.suptitle(plot_title, y=1.0)
        if plots_directory is not None:
            plt.savefig(plots_directory+"PCAL_U.png")
        plt.close()

        # Plotting the DC cuts
        region1_low_bin, region1_high_bin, region1_num_bins = (-150, 150), (-150, 150), (250, 250)
        region2_low_bin, region2_high_bin, region2_num_bins = (-200, 200), (-200, 200), (250, 250)
        region3_low_bin, region3_high_bin, region3_num_bins = (-250, 250), (-250, 250), (250, 250)

        fig, axs = plt.subplots(1, 3, figsize=(18,6))
        _, _, _, mesh = axs[0].hist2d(
            np.array(events["reconstructed"]["DC_region1_x"]),
            np.array(events["reconstructed"]["DC_region1_y"]),
            bins = region1_num_bins,
            range=(region1_low_bin, region1_high_bin),
            norm=colors.LogNorm(),
        )
        axs[0].set_ylabel("y (cm)")
        axs[0].set_xlabel("x (cm)")
        axs[0].set_title("DC Region 1")

        divider = make_axes_locatable(axs[0])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = fig.colorbar(mesh, cax=cax)

        _, _, _, mesh = axs[1].hist2d(
            np.array(events["reconstructed"]["DC_region2_x"]),
            np.array(events["reconstructed"]["DC_region2_y"]),
            bins = region2_num_bins,
            range=(region2_low_bin, region2_high_bin),
            norm=colors.LogNorm(),
        )
        axs[1].set_ylabel("y (cm)")
        axs[1].set_xlabel("x (cm)")
        axs[1].set_title("DC Region 2")

        divider = make_axes_locatable(axs[1])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = fig.colorbar(mesh, cax=cax)

        _, _, _, mesh = axs[2].hist2d(
            np.array(events["reconstructed"]["DC_region3_x"]),
            np.array(events["reconstructed"]["DC_region3_y"]),
            bins = region3_num_bins,
            range=(region3_low_bin, region3_high_bin),
            norm=colors.LogNorm(),
        )
        axs[2].set_ylabel("y (cm)")
        axs[2].set_xlabel("x (cm)")
        axs[2].set_title("DC Region 3")

        divider = make_axes_locatable(axs[2])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = fig.colorbar(mesh, cax=cax)

        plt.tight_layout()
        if plot_title is not None:
            plt.suptitle(plot_title, y=1.0)
        if plots_directory is not None:
            plt.savefig(plots_directory+"DC_before_fiducialcuts.png")
        plt.close()

        # Plot of chi2/NDF
        distance_to_edge_low_bin = 0
        distance_to_edge_high_bin = 20
        distance_to_edge_num_bins = 25
        bins = np.linspace(distance_to_edge_low_bin, distance_to_edge_high_bin, distance_to_edge_num_bins + 1)
        bin_centers = 0.5 * (bins[:-1] + bins[1:])

        edge_cut_values = {"region1":DC_region1_cut, "region2":DC_region2_cut, "region3":DC_region3_cut}
        fig, axs = plt.subplots(1, 3, figsize=(24, 7))

        for region_i in range(3):
            max_value = 0
            for sector in range(num_sectors):
                sector_mask = np.array(events["reconstructed"]["sector"] == sector + 1)
            
                distance_to_edge = np.array(events["reconstructed"][f"DC_region{region_i+1}_edge"][sector_mask])
                chi2 = np.array(events["reconstructed"]["chi2"][sector_mask])
                ndf = np.array(events["reconstructed"]["NDF"][sector_mask])
                chi2_per_ndf = chi2 / ndf
                bin_indices = np.digitize(distance_to_edge, bins) - 1
            
                bin_means = []
                for i in range(distance_to_edge_num_bins):
                    values_in_bin = chi2_per_ndf[bin_indices == i]
                    if len(values_in_bin) > 0:
                        bin_means.append(np.mean(values_in_bin))
                    else:
                        bin_means.append(np.nan)
            
                axs[region_i].scatter(bin_centers, bin_means, label=f"Sector {sector + 1}")
                bin_means = np.array(bin_means)[~np.isnan(bin_means)]
                max_bin_means = max(bin_means)
                if max_bin_means>max_value:
                    max_value=max_bin_means
            if edge_cut_values[f"region{region_i+1}"] is not None:
                axs[region_i].vlines(edge_cut_values[f"region{region_i+1}"], 0, max_value, color='red')
            axs[region_i].set_title(f"DC Region {region_i+1}")
            axs[region_i].set_xlabel("Distance to Edge (cm)")
            axs[region_i].set_ylabel("Average χ²/NDF")
            axs[region_i].legend(ncols=2, loc='upper right', columnspacing=.8)
            axs[region_i].grid(True)
        plt.tight_layout()
        if plot_title is not None:
            plt.suptitle(plot_title, y=1.0)
        if plots_directory is not None:
            plt.savefig(plots_directory+"DC_chi2NDF.png")
        plt.close()
        
        # DC plot with fiducial cuts
        region1_low_bin, region1_high_bin, region1_num_bins = (-150, 150), (-150, 150), (250, 250)
        region2_low_bin, region2_high_bin, region2_num_bins = (-200, 200), (-200, 200), (250, 250)
        region3_low_bin, region3_high_bin, region3_num_bins = (-250, 250), (-250, 250), (250, 250)

        fig, axs = plt.subplots(1, 3, figsize=(18,6))
        _, _, _, mesh = axs[0].hist2d(
            np.array(events["reconstructed"]["DC_region1_x"][DC_fiducial_mask]),
            np.array(events["reconstructed"]["DC_region1_y"][DC_fiducial_mask]),
            bins = region1_num_bins,
            range=(region1_low_bin, region1_high_bin),
            norm=colors.LogNorm(),
        )
        axs[0].set_ylabel("y (cm)")
        axs[0].set_xlabel("x (cm)")
        axs[0].set_title("DC Region 1")

        divider = make_axes_locatable(axs[0])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = fig.colorbar(mesh, cax=cax)

        _, _, _, mesh = axs[1].hist2d(
            np.array(events["reconstructed"]["DC_region2_x"][DC_fiducial_mask]),
            np.array(events["reconstructed"]["DC_region2_y"][DC_fiducial_mask]),
            bins = region2_num_bins,
            range=(region2_low_bin, region2_high_bin),
            norm=colors.LogNorm(),
        )
        axs[1].set_ylabel("y (cm)")
        axs[1].set_xlabel("x (cm)")
        axs[1].set_title("DC Region 2")

        divider = make_axes_locatable(axs[1])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = fig.colorbar(mesh, cax=cax)

        _, _, _, mesh = axs[2].hist2d(
            np.array(events["reconstructed"]["DC_region3_x"][DC_fiducial_mask]),
            np.array(events["reconstructed"]["DC_region3_y"][DC_fiducial_mask]),
            bins = region3_num_bins,
            range=(region3_low_bin, region3_high_bin),
            norm=colors.LogNorm(),
        )
        axs[2].set_ylabel("y (cm)")
        axs[2].set_xlabel("x (cm)")
        axs[2].set_title("DC Region 3")

        divider = make_axes_locatable(axs[2])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = fig.colorbar(mesh, cax=cax)

        plt.tight_layout()
        if plot_title is not None:
            plt.suptitle(plot_title, y=1.0)
        if plots_directory is not None:
            plt.savefig(plots_directory+"DC_after_fiducialcuts.png")
        plt.close()
    
    fiducial_cuts = (PCAL_fiducial_mask) & (DC_fiducial_mask)

    if log_file is not None:
        with open(log_file, "a") as f:
            f.write(f"Have {ak.sum(PCAL_fiducial_mask)/number_of_initial_electrons} fraction of events after PCAL fiducial cuts\n")
            f.write(f"Have {ak.sum(DC_fiducial_mask)/number_of_initial_electrons} fraction of events after DC fiducial cuts\n")
            f.write(f"Have {ak.sum(fiducial_cuts)/number_of_initial_electrons} fraction of events after PCAL and DC fiducial cuts\n")
            f.write(f"Have {ak.sum(DC_region1_mask)/number_of_initial_electrons} fraction of events after DC region 1 fiducial cut\n")
            f.write(f"Have {ak.sum(DC_region2_mask)/number_of_initial_electrons} fraction of events after DC region 2 fiducial cut\n")
            f.write(f"Have {ak.sum(DC_region3_mask)/number_of_initial_electrons} fraction of events after DC region 3 fiducial cut\n")

    events["pass_reco"] = (fiducial_cuts) & (events["pass_reco"])

    print(f"Have {ak.sum(events['pass_reco'])} events after fiducial cuts")
    if log_file is not None:
        with open(log_file, "a") as f:
            f.write(f"Have {ak.sum(events['pass_reco'])} pass reco events after fiducial cuts\n")
        with open(log_file, "a") as f:
            f.write(f"Have {ak.sum(events['pass_reco'])} fraction of events after PCAL and DC fiducial cuts and kinematic cuts\n")
    
    events["pass_fiducial_and_kinematic"] = events["pass_reco"]
    return events

def apply_partial_sampling_fraction_cut(events, is_simulation = False, save_plots = True, plots_directory = None, plot_title = None, log_file = None, number_of_initial_electrons = None):

    def ecin_epcal_cut(ecin, is_simulation):
        if is_simulation:
            return (-.22/.18)*ecin + .22
        else:
            return (-.22/.15)*ecin + .22
    ECIN_SF = np.array(events["reconstructed"]["E_ECIN"]/events["reconstructed"]["p"])
    partial_SF_mask = np.array(events["reconstructed"]["E_PCAL"]/events["reconstructed"]["p"]) > ecin_epcal_cut(ECIN_SF, is_simulation)
    partial_SF_mask[events["reconstructed"]["p"] < 4.5] = True

    if save_plots:
        events_without_partialsampling_cut = events[events["pass_fiducial_and_kinematic"]]
    # Updating the pass_reco mask to include the partial sampling fraction cut
    events["pass_reco"] = (events["pass_reco"]) & (partial_SF_mask)
    if save_plots:
        events_with_partialsampling_cut = events[(events["pass_fiducial_and_kinematic"]) & (partial_SF_mask)]

    if save_plots:
        fig, axs = plt.subplots(3, 2, figsize=(18, 18))
        axs = axs.flatten()
        for sector in range(num_sectors):
            sector_cut = events_without_partialsampling_cut["reconstructed"]["sector"]==(sector+1)
            if len(np.array(events_without_partialsampling_cut["reconstructed"]["E_ECIN"]/events_without_partialsampling_cut["reconstructed"]["p"])[sector_cut])==0:
                continue
            hist, ecin_bins, epcal_bins, mesh = axs[sector].hist2d(
                np.array(events_without_partialsampling_cut["reconstructed"]["E_ECIN"]/events_without_partialsampling_cut["reconstructed"]["p"])[sector_cut], 
                np.array(events_without_partialsampling_cut["reconstructed"]["E_PCAL"]/events_without_partialsampling_cut["reconstructed"]["p"])[sector_cut],
                bins=(100,100),
                range=[(0,.2), (0,.25)],
                norm=colors.LogNorm())
            axs[sector].set_xlabel("$E_{ECIN}$/p")
            axs[sector].set_ylabel("$E_{PCAL}$/p")
            axs[sector].set_title(f"Sector {sector+1}") 
            divider = make_axes_locatable(axs[sector])
            cax = divider.append_axes("right", size="5%", pad=0.05)
            cbar = fig.colorbar(mesh, cax=cax)
            axs[sector].plot(ecin_bins.tolist(), ecin_epcal_cut(np.array(ecin_bins), is_simulation), color='red')
        fig.tight_layout()
        if plot_title is not None:
            plt.suptitle(plot_title, y=1.0)
        if plots_directory is not None:
            plt.savefig(plots_directory+"partial_sampling_fraction_beforecut.png")
        plt.close()
        
        fig, axs = plt.subplots(3, 2, figsize=(18, 18))
        axs = axs.flatten()
        for sector in range(num_sectors):
            sector_cut = events_with_partialsampling_cut["reconstructed"]["sector"]==(sector+1)
            if len(np.array(events_with_partialsampling_cut["reconstructed"]["E_ECIN"]/events_with_partialsampling_cut["reconstructed"]["p"])[sector_cut])==0:
                continue
            hist, ecin_bins, epcal_bins, mesh = axs[sector].hist2d(
                np.array(events_with_partialsampling_cut["reconstructed"]["E_ECIN"]/events_with_partialsampling_cut["reconstructed"]["p"])[sector_cut], 
                np.array(events_with_partialsampling_cut["reconstructed"]["E_PCAL"]/events_with_partialsampling_cut["reconstructed"]["p"])[sector_cut],
                bins=(100,100),
                range=[(0,.2), (0,.25)],
                norm=colors.LogNorm())
            axs[sector].set_xlabel("$E_{ECIN}$/p")
            axs[sector].set_ylabel("$E_{PCAL}$/p")
            axs[sector].set_title(f"Sector {sector+1}") 
            divider = make_axes_locatable(axs[sector])
            cax = divider.append_axes("right", size="5%", pad=0.05)
            cbar = fig.colorbar(mesh, cax=cax)
            axs[sector].plot(ecin_bins.tolist(), ecin_epcal_cut(np.array(ecin_bins), is_simulation), color='red')
        fig.tight_layout()
        if plot_title is not None:
            plt.suptitle(plot_title, y=1.0)
        if plots_directory is not None:
            plt.savefig(plots_directory+"partial_sampling_fraction_aftercut.png")
        plt.close()
    
    print(f"Have {ak.sum(events['pass_reco'])} events after partial SF cuts")
    if log_file is not None:
        with open(log_file, "a") as f:
            f.write(f"Have {ak.sum(events['pass_reco'])} pass reco events after partial SF cuts\n")
            f.write(f"Have {ak.sum(partial_SF_mask)/number_of_initial_electrons} fraction of events after partial SF cuts\n")
            f.write(f"Have {ak.sum((events['pass_fiducial_and_kinematic']) & (partial_SF_mask))/number_of_initial_electrons} fraction of events after fiducial,kinematic, and partial SF cuts\n")
    events["pass_partial_SF"] = partial_SF_mask
    return events

def gaus(x, a, mu, sigma):
    return a*np.exp(-(x-mu)**2/(2*sigma*sigma))

def sf_gaussians_by_sector(sampling_fractions_in_sector,
                           xaxis_in_sector,
                           xaxis_bins_in_sector,
                           sector_number,
                           SF_bins,
                           xaxis_name,
                           save_plots = True,
                           plots_directory = None,
                           plot_title = None):
    sf_fit_data = {
        "bin_low": [],
        "bin_high": [],
        "bin_center": [],
        "mu": [],
        "sigma": []
    }
    low_sf_bin, high_sf_bin = SF_bins[0], SF_bins[1]
    fig, axs = plt.subplots(10, 10, figsize=(45, 55))
    fig.subplots_adjust(hspace=0.1, wspace=0.1)
    axs = axs.flatten()
    for i, lower_bin_edge in enumerate(xaxis_bins_in_sector):
        if i == len(xaxis_bins_in_sector)-1:
            break
        upper_bin_edge = xaxis_bins_in_sector[i+1]
        xaxis_bin_mask = (xaxis_in_sector>lower_bin_edge) & (xaxis_in_sector<upper_bin_edge)
        sector_sf_mask = (sampling_fractions_in_sector[xaxis_bin_mask] > .2) & (sampling_fractions_in_sector[xaxis_bin_mask] <.27)
        
        counts, bins, _ = axs[i].hist(sampling_fractions_in_sector[xaxis_bin_mask], bins=100, range=(low_sf_bin, high_sf_bin))
        
        bin_centers = (bins[:-1] + bins[1:])/2
        sf_mask = (bin_centers>.2) & (bin_centers<.3)
        popt, pcov = curve_fit(
            gaus,
            bin_centers[sf_mask],
            counts[sf_mask],
            p0=(len(sampling_fractions_in_sector[xaxis_bin_mask][sector_sf_mask]),
                np.mean(sampling_fractions_in_sector[xaxis_bin_mask][sector_sf_mask]),
                np.std(sampling_fractions_in_sector[xaxis_bin_mask][sector_sf_mask]))
        )
        sf_fit_data["bin_low"].append(lower_bin_edge)
        sf_fit_data["bin_high"].append(upper_bin_edge)
        sf_fit_data["bin_center"].append((upper_bin_edge+lower_bin_edge)/2)
        sf_fit_data["mu"].append(popt[1])
        sf_fit_data["sigma"].append(popt[2])
        axs[i].plot(bin_centers, gaus(bin_centers, *popt))
        axs[i].set_xlabel("SF", fontsize=10)
        axs[i].set_title(f"{round(lower_bin_edge,3)} GeV < {xaxis_name} < {round(upper_bin_edge,3)} GeV", fontsize=10)
        axs[i].tick_params(axis='both', which='major', labelsize=10)
    fig.tight_layout()
    
    if save_plots:
        if plot_title is not None:
            fig.suptitle(f"Sector {sector_number}", y = 1.01)
        if plots_directory is not None:
            plt.savefig(plots_directory+f"sector{sector_number}_gaussian_fit.png")
    plt.close()
    sf_fit_data_df = pd.DataFrame(sf_fit_data)
    return sf_fit_data_df

def apply_sampling_fraction_cut(events, save_plots = True, plots_directory = None, plot_title = None, log_file = None, number_of_initial_electrons = None):

    low_edep_bin = .6
    high_edep_bin = 1.6
    low_sf_bin = .1
    high_sf_bin = .35
    
    events["reconstructed"] = ak.with_field(
        events["reconstructed"],
        events["reconstructed"]["E_PCAL"] + events["reconstructed"]["E_ECIN"] + events["reconstructed"]["E_ECOUT"],
        "total_ecal_energy"
    )

    electrons = events["reconstructed"]
    fig, axs = plt.subplots(3, 2, figsize=(18, 18))
    axs = axs.flatten()
    sampling_fraction_by_sector = []
    edep_by_sector = []
    edep_bins_by_sector = []
    for sector in range(num_sectors):
        sector_cut = (electrons["sector"]==(sector+1)) & (events["pass_fiducial_and_kinematic"])
        total_ecal_energy = np.array(electrons["total_ecal_energy"][sector_cut])
        sampling_fraction = np.array(electrons["SF"][sector_cut])
        sampling_fraction_by_sector.append(sampling_fraction)
        edep_by_sector.append(total_ecal_energy)
        
        _, edep_bins, _, mesh= axs[sector].hist2d(
            total_ecal_energy,
            sampling_fraction,
            bins=(100,100),
            range=[(low_edep_bin, high_edep_bin),(low_sf_bin, high_sf_bin)],
            norm=colors.LogNorm())
        edep_bins_by_sector.append(edep_bins)
        axs[sector].set_xlabel("$E_{dep}$ (GeV)")
        axs[sector].set_ylabel("$(E_{PCAL}+E_{ECIN}+E_{ECOUT})/P$")
        axs[sector].set_title(f"Sector {sector+1}") 
        divider = make_axes_locatable(axs[sector])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = fig.colorbar(mesh, cax=cax)
    fig.tight_layout()
    if save_plots:
        if plot_title is not None:
                plt.suptitle(plot_title, y=1.0)
        if plots_directory is not None:
            plt.savefig(plots_directory+"sector_SF_without_fit.png")
    plt.close()

    sf_fit_data_df_by_sector = []
    popt_mu_by_sector, popt_sigma_by_sector = [], []
    def sf_fit_function(x, a, b, c, d):
        return a + b*x  + c*(x*x) + d*(x*x*x)
    # In each sector, fitting the sampling fraction in bins of Edep for the events that pass reconstruction
    for sector in range(num_sectors):
        sector_number = sector + 1
        print(f"Fitting SF vs. Edep for sector {sector_number}")
        sf_df = sf_gaussians_by_sector(sampling_fraction_by_sector[sector],
                                    edep_by_sector[sector],
                                    edep_bins_by_sector[sector],
                                    sector_number,
                                    SF_bins = (low_sf_bin, high_sf_bin),
                                    xaxis_name="$E_{dep}$",
                                    save_plots = save_plots,
                                    plots_directory = plots_directory,
                                    plot_title = plot_title)
        sf_fit_data_df_by_sector.append(sf_df)

        # Fitting the mu and sigma vs momentum for each sector
        fig, axs = plt.subplots(1, 2, figsize=(15, 6))
        fig.subplots_adjust(hspace=0.1, wspace=0.3)
        sf_fit_data_df = sf_fit_data_df_by_sector[sector]
        axs[0].scatter(sf_fit_data_df["bin_center"].tolist(), sf_fit_data_df["mu"].tolist())
        axs[0].set_xlabel("bin center (GeV)")
        axs[0].set_ylabel("SF $\mu$")
        popt_mu, pcov_mu = curve_fit(sf_fit_function, sf_fit_data_df["bin_center"].tolist(), sf_fit_data_df["mu"].tolist(), p0=(.2, .001, .00001, .00001))
        axs[0].plot(sf_fit_data_df["bin_center"].tolist(), sf_fit_function(np.array(sf_fit_data_df["bin_center"].tolist()), *popt_mu), color='red')
        axs[1].scatter(sf_fit_data_df["bin_center"].tolist(), sf_fit_data_df["sigma"].tolist())
        axs[1].set_xlabel("bin center (GeV)")
        axs[1].set_ylabel("SF $\sigma$")
        popt_sigma, pcov_sigma = curve_fit(sf_fit_function, sf_fit_data_df["bin_center"].tolist(), sf_fit_data_df["sigma"].tolist(), p0=(.002, .001, .00001, .00001))
        axs[1].plot(sf_fit_data_df["bin_center"].tolist(), sf_fit_function(np.array(sf_fit_data_df["bin_center"].tolist()), *popt_sigma), color='red')
        if save_plots:
            if plot_title is not None:
                    plt.suptitle(plot_title+f",Sector {sector_number}", y=1.0)
            if plots_directory is not None:
                plt.savefig(plots_directory+f"SF_mu_fits_sector{sector_number}.png")
        plt.close()
        popt_mu_by_sector.append(popt_mu)
        popt_sigma_by_sector.append(popt_sigma)
        
        
    fig, axs = plt.subplots(3, 2, figsize=(18, 18))
    axs = axs.flatten()

    # Plotting the SF vs edep for events that pass reco before SF cut and the SF cut curves
    if save_plots:
        for sector in range(num_sectors):
            # Getting the fit parameters for our given sector
            popt_mu = popt_mu_by_sector[sector]
            popt_sigma = popt_sigma_by_sector[sector]
            
            hist, edep_bins, sf_bins, mesh= axs[sector].hist2d(
                edep_by_sector[sector],
                sampling_fraction_by_sector[sector],
                bins=(100,100),
                range=[(0.3,2.1),(.15,.32)],
                norm=colors.LogNorm()
            )

            edep_bin_centers = (edep_bins[:-1] + edep_bins[1:])/2
            
            axs[sector].set_xlabel("$E_{dep}$ (GeV)")
            axs[sector].set_ylabel("$(E_{PCAL}+E_{ECIN}+E_{ECOUTT})/P$")
            axs[sector].set_title(f"Sector {sector+1}") 
            divider = make_axes_locatable(axs[sector])
            cax = divider.append_axes("right", size="5%", pad=0.05)
            cbar = fig.colorbar(mesh, cax=cax)

            axs[sector].plot(edep_bin_centers.tolist(), sf_fit_function(np.array(edep_bin_centers.tolist()), *popt_mu), color='black')
            axs[sector].plot(
                edep_bin_centers.tolist(),
                sf_fit_function(np.array(edep_bin_centers.tolist()), *popt_mu) + 3.5*sf_fit_function(np.array(edep_bin_centers.tolist()), *popt_sigma),
                color='red'
            )
            axs[sector].plot(
                edep_bin_centers.tolist(),
                sf_fit_function(np.array(edep_bin_centers.tolist()), *popt_mu) - 3.5*sf_fit_function(np.array(edep_bin_centers.tolist()), *popt_sigma),
                color='red'
            )

        fig.tight_layout()
        if plot_title is not None:
                plt.suptitle(plot_title, y=1.0)
        if plots_directory is not None:
            plt.savefig(plots_directory+"sector_SF_with_fit.png")
        plt.close()

    new_pass_reco_mask = np.ones(len(events["pass_reco"]), dtype=bool)
    SF_mask = np.ones(len(events["pass_reco"]), dtype=bool)
    for sector in range(num_sectors):
        sector_mask = (electrons["sector"]==(sector+1))

        popt_mu = popt_mu_by_sector[sector]
        popt_sigma = popt_sigma_by_sector[sector]
        edep_in_sector = electrons["total_ecal_energy"][sector_mask]
        sampling_fraction_in_sector = electrons["SF"][sector_mask]
        fit_mu = sf_fit_function(edep_in_sector, *popt_mu)
        fit_sigma = sf_fit_function(edep_in_sector, *popt_sigma)
        if log_file is not None:
            with open(log_file, "a") as f:
                f.write(f"Sector {sector+1} SF cut parameters:\n")
                f.write(f"mu fit parameters: {popt_mu}\n")
                f.write(f"sigma fit parameters: {popt_sigma}\n")
        SF_mask[sector_mask] = (sampling_fraction_in_sector < (fit_mu + 3.5*fit_sigma)) & (sampling_fraction_in_sector > (fit_mu - 3.5*fit_sigma))
        new_pass_reco_mask[sector_mask] = (events["pass_reco"][sector_mask]) & (sampling_fraction_in_sector < (fit_mu + 3.5*fit_sigma)) & (sampling_fraction_in_sector > (fit_mu - 3.5*fit_sigma))

    events["pass_reco"] = new_pass_reco_mask
    print(f"Have {ak.sum(events['pass_reco'])} events after SF cuts")
    if log_file is not None:
        with open(log_file, "a") as f:
            f.write(f"Have {ak.sum(events['pass_reco'])} pass reco events after SF cuts\n")
            f.write(f"Have {ak.sum(SF_mask)/number_of_initial_electrons} fraction of events after SF cuts\n")
            f.write(f"Have {ak.sum((events['pass_fiducial_and_kinematic']) & (SF_mask))/number_of_initial_electrons} fraction of events after fiducial,kinematic, and SF cuts\n")
            f.write(f"Have {ak.sum((events['pass_partial_SF']) & (SF_mask))/number_of_initial_electrons} fraction of events after partial and SF cuts\n")
            f.write(f"Have {ak.sum((events['pass_fiducial_and_kinematic']) & (events['pass_partial_SF']) & (SF_mask))/ak.sum((events['pass_fiducial_and_kinematic']))} fraction of events that pass fiducial and kinematic cuts after partial and SF cuts\n")
    return events

def apply_target_selection(events, solid_target_name, save_plots = True, plots_directory = None, plot_title = None, log_file = None, number_of_initial_electrons = None):

    def double_gaussian(x, amp1, mean1, sigma1, amp2, mean2, sigma2):
        return amp1 * np.exp( -(x - mean1)**2 / (2*sigma1) ) + \
            amp2 * np.exp( -(x - mean2)**2 / (2*sigma2) )
    
    if save_plots:
        fig, axs = plt.subplots(3, 2, figsize=(18, 18))
        axs = axs.flatten()
    num_sectors = 6
    z_fit_parameters_all_sectors = []
    
    electrons = events["reconstructed"]
    for sector in range(num_sectors):

        sector_cut = (electrons["sector"]==(sector+1)) & (events["pass_reco"])
        
        vertex_z = np.array(electrons["v_z"][sector_cut])
        vertex_z_counts, vertex_z_bins = np.histogram(
            vertex_z,
            bins=100,
            range=(-12,5),
        )
        vertex_z_bin_centers = (vertex_z_bins[:-1] + vertex_z_bins[1:])/2
        lower_fit_range = -12
        upper_fit_range = 12

        amplitude_guess = np.max(vertex_z_counts)
        mean_z_guess = ak.mean(vertex_z)
        std_z_guess = ak.std(vertex_z)
        
        z_fit_parameters, z_fit_covariance = curve_fit(
            double_gaussian,
            vertex_z_bin_centers,
            vertex_z_counts, 
            p0=(amplitude_guess, -7, 2.5, amplitude_guess, -1.5, 1.5))
        y_fit = double_gaussian(vertex_z_bin_centers, *z_fit_parameters)
        z_fit_parameters_all_sectors.append(z_fit_parameters)

        if save_plots:
            axs[sector].hist(
                vertex_z,
                bins=100,
                range=(-12,5),
                histtype='step'
            )
            axs[sector].plot(vertex_z_bin_centers, y_fit, color='r')
            axs[sector].set_xlabel("$v_{z}$ (cm)")
            axs[sector].set_title(f"Sector {sector+1}") 
    if save_plots:
        fig.tight_layout()
        if plot_title is not None:
            plt.suptitle(plot_title, y=1.0)
        if plots_directory is not None:
            plt.savefig(plots_directory+"zvertex_fits.png")

    deuterium_mean_by_sector, deuterium_sigma_by_sector = [], []
    solid_mean_by_sector, solid_sigma_by_sector = [], []

    if save_plots:
        fig, axs = plt.subplots(3, 2, figsize=(18, 18))
        axs = axs.flatten()
    for sector in range(num_sectors):
        z_fit_parameters = z_fit_parameters_all_sectors[sector]
        sector_cut = (electrons["sector"]==(sector+1)) & (events["pass_reco"])
        
        vertex_z = np.array(electrons["v_z"][sector_cut])
        vertex_z_counts, vertex_z_bins = np.histogram(
            vertex_z,
            bins=100,
            range=(-12,5),
        )
        vertex_z_bin_centers = (vertex_z_bins[:-1] + vertex_z_bins[1:])/2
        deuterium_z_mean = min(z_fit_parameters[1], z_fit_parameters[4])
        if deuterium_z_mean == z_fit_parameters[1]:
            deuterium_z_sigma = z_fit_parameters[2]
            solid_z_mean, solid_z_sigma = z_fit_parameters[4], z_fit_parameters[5]
        elif deuterium_z_mean == z_fit_parameters[4]:
            deuterium_z_sigma = z_fit_parameters[5]
            solid_z_mean, solid_z_sigma = z_fit_parameters[1], z_fit_parameters[2]
        
        deuterium_cut = (vertex_z > (deuterium_z_mean - 3 * deuterium_z_sigma) ) &\
                        (vertex_z < (deuterium_z_mean + 3 * deuterium_z_sigma) )
        solid_cut     = (vertex_z > (solid_z_mean - 5 * solid_z_sigma) ) &\
                        (vertex_z < (solid_z_mean + 5 * solid_z_sigma) )
        deuterium_mean_by_sector.append(deuterium_z_mean)
        deuterium_sigma_by_sector.append(deuterium_z_sigma)
        solid_mean_by_sector.append(solid_z_mean)
        solid_sigma_by_sector.append(solid_z_sigma)
        
        if save_plots:
            axs[sector].hist(
                vertex_z,
                bins=100,
                range=(-12,5),
                histtype='step'
            )
            axs[sector].hist(vertex_z[deuterium_cut],
                bins = 100,
                range=(-12, 5),
                color='b',
                label="LD2",
                alpha=.8)
            axs[sector].hist(vertex_z[solid_cut],
                    bins = 100,
                    range=(-12, 5),
                    color='r',
                    label=solid_target_name,
                    alpha=.8)
            
            axs[sector].set_xlabel("$v_{z}$ (cm)")
            axs[sector].set_title(f"Sector {sector+1}") 
            axs[sector].legend(loc='upper left')
    if save_plots:
        fig.tight_layout()
        if plot_title is not None:
            plt.suptitle(plot_title, y=1.0)
        if plots_directory is not None:
            plt.savefig(plots_directory+"target_selections.png")

    deuterium_mask = np.ones(len(events["pass_reco"]), dtype=bool)
    solid_mask = np.ones(len(events["pass_reco"]), dtype=bool)
    for sector in range(num_sectors):
        sector_mask = (electrons["sector"]==(sector+1))
        vertex_z_in_sector = electrons["v_z"][sector_mask]
        deuterium_mask_in_sector = (vertex_z_in_sector > (deuterium_mean_by_sector[sector] - 3*deuterium_sigma_by_sector[sector]) ) &\
                         (vertex_z_in_sector < (deuterium_mean_by_sector[sector] + 3 * deuterium_sigma_by_sector[sector]) )
        solid_mask_in_sector = (vertex_z_in_sector > (solid_mean_by_sector[sector] - 5*solid_sigma_by_sector[sector]) ) &\
                         (vertex_z_in_sector < (solid_mean_by_sector[sector] + 5 * solid_sigma_by_sector[sector]) )
        

        deuterium_mask[sector_mask] = (events["pass_reco"][sector_mask]) & (deuterium_mask_in_sector)
        solid_mask[sector_mask] = (events["pass_reco"][sector_mask]) & (solid_mask_in_sector)
        if log_file is not None:
            with open(log_file, "a") as f:
                f.write(f"Sector {sector+1} target selection parameters:\n")
                f.write(f"Deuterium mean (cm): {deuterium_mean_by_sector[sector]}, sigma (cm): {deuterium_sigma_by_sector[sector]}\n")
                f.write(f"{solid_target_name} mean (cm): {solid_mean_by_sector[sector]}, sigma (cm): {solid_sigma_by_sector[sector]}\n")
                f.write(f"Deuterium cut (cm): {deuterium_mean_by_sector[sector] - 3*deuterium_sigma_by_sector[sector]} to {deuterium_mean_by_sector[sector] + 3*deuterium_sigma_by_sector[sector]}\n")
                f.write(f"{solid_target_name} cut (cm): {solid_mean_by_sector[sector] - 5*solid_sigma_by_sector[sector]} to {solid_mean_by_sector[sector] + 5*solid_sigma_by_sector[sector]}\n")

    if log_file is not None:
        with open(log_file, "a") as f:
            f.write(f"Have {ak.sum((deuterium_mask) | (solid_mask))/ak.sum(events['pass_reco'])} fraction of events after target selection cuts\n")
    events["pass_reco"] = (deuterium_mask) | (solid_mask)
    target = np.empty(len(events), dtype=object)
    target[deuterium_mask] = "LD2"
    target[solid_mask] = solid_target_name
    target[(~deuterium_mask)&(~solid_mask)] = "None"

    events["reconstructed"] = ak.with_field(
        events["reconstructed"],
        target.tolist(),
        "target"
    )

    print(f"Have {ak.sum(events['pass_reco'])} events after target selection cuts")
    if log_file is not None:
        with open(log_file, "a") as f:
            f.write(f"Have {ak.sum(events['pass_reco'])} events after target selection cuts\n")

    return events