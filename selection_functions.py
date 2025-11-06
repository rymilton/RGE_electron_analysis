import matplotlib.pyplot as plt
import matplotlib.colors as colors
from mpl_toolkits.axes_grid1 import make_axes_locatable
import mplhep as hep 
hep.style.use(hep.style.CMS)
import numpy as np 

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
def apply_kinematic_cuts(events, kinematic_cuts):
    mask = np.ones(len(events), dtype=bool)
    for cut in kinematic_cuts:
        variable_name, operation, cut_value = cut.split()
        if operation not in array_operator_dict:
            raise ValueError(f"Unsupported operation: {operation}")
        
        mask = (mask) & (array_operator_dict[operation](events["reconstructed"][variable_name], float(cut_value)))
    masked_events = events[mask]
    print(f"Have {len(masked_events)} events after kinematic cuts")
    return masked_events

# Applying fiducial cuts to each electron based on ELECTRON_FIDUCIAL_CUTS in the config file
def apply_fiducial_cuts(events, fiducial_cuts, save_plots = True, plots_directory = None, plot_title = None):
    

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
            DC_fiducial_mask = (DC_fiducial_mask) & (array_operator_dict[operation](events["reconstructed"][variable_name], float(cut_value)))
            DC_region1_cut = float(cut_value)
        elif variable_name == "DC_region2_edge":
            DC_fiducial_mask = (DC_fiducial_mask) & (array_operator_dict[operation](events["reconstructed"][variable_name], float(cut_value)))
            DC_region2_cut = float(cut_value)
        elif variable_name == "DC_region3_edge":
            DC_fiducial_mask = (DC_fiducial_mask) & (array_operator_dict[operation](events["reconstructed"][variable_name], float(cut_value)))
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
    
    fiducial_cuts = (PCAL_fiducial_mask) & (DC_fiducial_mask)
    masked_events = events[fiducial_cuts]
    print(f"Have {len(masked_events)} events after fiducial cuts")
    return masked_events

def apply_partial_sampling_fraction_cut(events, is_simulation = False, save_plots = True, plots_directory = None, plot_title = None):

    def ecin_epcal_cut(ecin, is_simulation):
        if is_simulation:
            return (-.22/.18)*ecin + .22
        else:
            return (-.22/.15)*ecin + .22
    ECIN_SF = np.array(events["reconstructed"]["E_ECIN"]/events["reconstructed"]["p"])
    partial_SF_mask = np.array(events["reconstructed"]["E_PCAL"]/events["reconstructed"]["p"]) > ecin_epcal_cut(ECIN_SF, is_simulation)
    partial_SF_mask[events["reconstructed"]["p"] < 4.5] = True
    masked_events = events[partial_SF_mask]

    if save_plots:
        fig, axs = plt.subplots(3, 2, figsize=(18, 18))
        axs = axs.flatten()
        for sector in range(num_sectors):
            sector_cut = events["reconstructed"]["sector"]==(sector+1)
            if len(np.array(events["reconstructed"]["E_ECIN"]/events["reconstructed"]["p"])[sector_cut])==0:
                continue
            hist, ecin_bins, epcal_bins, mesh = axs[sector].hist2d(
                np.array(events["reconstructed"]["E_ECIN"]/events["reconstructed"]["p"])[sector_cut], 
                np.array(events["reconstructed"]["E_PCAL"]/events["reconstructed"]["p"])[sector_cut],
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
        
        fig, axs = plt.subplots(3, 2, figsize=(18, 18))
        axs = axs.flatten()
        for sector in range(num_sectors):
            sector_cut = masked_events["reconstructed"]["sector"]==(sector+1)
            if len(np.array(masked_events["reconstructed"]["E_ECIN"]/masked_events["reconstructed"]["p"])[sector_cut])==0:
                continue
            hist, ecin_bins, epcal_bins, mesh = axs[sector].hist2d(
                np.array(masked_events["reconstructed"]["E_ECIN"]/masked_events["reconstructed"]["p"])[sector_cut], 
                np.array(masked_events["reconstructed"]["E_PCAL"]/masked_events["reconstructed"]["p"])[sector_cut],
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
    
    print(f"Have {len(masked_events)} events after kinematic cuts")
    return masked_events
