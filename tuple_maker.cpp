#include <TFile.h>
#include <TTree.h>
#include <TTreeReader.h>
#include <TTreeReaderArray.h>
#include <TTreeReaderValue.h>
#include <TString.h>
#include <iostream>
#include <cmath> // for std::sqrt, std::atan2
#include <vector>

void tuple_maker(
    TString input_dir,
    TString input_file,
    TString output_dir,
    TString output_file,
    Bool_t save_MC
)
{
    if (!input_dir.EndsWith("/")) input_dir += "/";
    std::cout << "Opening file " << input_dir + input_file << std::endl;
    TFile* inFile = TFile::Open(input_dir + input_file);
    if (!inFile || inFile->IsZombie()) {
        std::cerr << "Error opening file " << input_dir + input_file << std::endl;
        return;
    }

    TTree* tree = (TTree*)inFile->Get("data");
    TTreeReader reader(tree);

    // Input branches (arrays)
    TTreeReaderArray<double> particle_pid_branch(reader, "REC::Particle::pid");
    TTreeReaderArray<double> particle_vx_branch(reader, "REC::Particle::vx");
    TTreeReaderArray<double> particle_vy_branch(reader, "REC::Particle::vy");
    TTreeReaderArray<double> particle_vz_branch(reader, "REC::Particle::vz");
    TTreeReaderArray<double> particle_vt_branch(reader, "REC::Particle::vt");
    TTreeReaderArray<double> particle_px_branch(reader, "REC::Particle::px");
    TTreeReaderArray<double> particle_py_branch(reader, "REC::Particle::py");
    TTreeReaderArray<double> particle_pz_branch(reader, "REC::Particle::pz");
    TTreeReaderArray<double> particle_charge_branch(reader, "REC::Particle::charge");
    TTreeReaderArray<double> particle_beta_branch(reader, "REC::Particle::beta");
    TTreeReaderArray<double> particle_chi2pid_branch(reader, "REC::Particle::chi2pid");
    TTreeReaderArray<double> particle_status_branch(reader, "REC::Particle::status");

    TTreeReaderArray<double> track_pindex_branch(reader, "REC::Track::pindex");
    TTreeReaderArray<double> track_charge_branch(reader, "REC::Track::q");
    TTreeReaderArray<double> track_sector_branch(reader, "REC::Track::sector");
    TTreeReaderArray<double> track_ndf_branch(reader, "REC::Track::ndf");
    TTreeReaderArray<double> track_chi2_branch(reader, "REC::Track::chi2");

    TTreeReaderArray<double> calo_pindex_branch(reader, "REC::Calorimeter::pindex");
    TTreeReaderArray<double> calo_layer_branch(reader, "REC::Calorimeter::layer");
    TTreeReaderArray<double> calo_energy_branch(reader, "REC::Calorimeter::energy");
    TTreeReaderArray<double> calo_lu_branch(reader, "REC::Calorimeter::lu");
    TTreeReaderArray<double> calo_lv_branch(reader, "REC::Calorimeter::lv");
    TTreeReaderArray<double> calo_lw_branch(reader, "REC::Calorimeter::lw");

    TTreeReaderArray<double> cherenkov_pindex_branch(reader, "REC::Cherenkov::pindex");
    TTreeReaderArray<double> cherenkov_detectorindex_branch(reader, "REC::Cherenkov::detector");
    TTreeReaderArray<double> cherenkov_photoelectrons_branch(reader, "REC::Cherenkov::nphe");

    TTreeReaderArray<double> trajectory_pindex_branch(reader, "REC::Traj::pindex");
    TTreeReaderArray<double> trajectory_detectorindex_branch(reader, "REC::Traj::detector");
    TTreeReaderArray<double> trajectory_layer_branch(reader, "REC::Traj::layer");
    TTreeReaderArray<double> trajectory_x_branch(reader, "REC::Traj::x");
    TTreeReaderArray<double> trajectory_y_branch(reader, "REC::Traj::y");
    TTreeReaderArray<double> trajectory_z_branch(reader, "REC::Traj::z");
    TTreeReaderArray<double> trajectory_edge_branch(reader, "REC::Traj::edge");

    TTreeReaderArray<double> fcupgated_branch(reader, "RUN::scaler::fcupgated");

    TTreeReaderArray<double> run_number_branch(reader, "RUN::config::run");
    TTreeReaderArray<double> event_number_branch(reader, "RUN::config::event");


    if (!output_dir.EndsWith("/")) {
        output_dir += "/";
    }

    std::cout << "Will save output to " << output_dir + output_file << std::endl;
    TFile* outFile = TFile::Open(output_dir + output_file, "RECREATE");
    TTree* outTree = new TTree("data", "Processed Data");
    TTree *outTree_meta = new TTree("meta", "Event level info");
    TTree* outTree_MC = new TTree("MC", "Monte Carlo Data");

    // Output branches: A vector of particles for each event
    std::vector<double> beta, chi2pid, px, py, pz, p, vt, vx, vy, vz, theta, phi;
    std::vector<int> charge, pid, status;
    std::vector<int> track_charge, sector, track_ndf;
    std::vector<double> track_chi2;

    std::vector<double> E_PCAL, E_ECIN, E_ECOUT;
    std::vector<double> PCAL_U, PCAL_V, PCAL_W;

    std::vector<int> Nphe_HTCC, Nphe_LTCC;

    std::vector<double> DC_region1_x, DC_region1_y, DC_region1_z, DC_region1_edge;
    std::vector<double> DC_region2_x, DC_region2_y, DC_region2_z, DC_region2_edge;
    std::vector<double> DC_region3_x, DC_region3_y, DC_region3_z, DC_region3_edge;

    double fcupgated;
    int run_number, event_number, num_tracks;

    std::vector<int> MC_pid;
    std::vector<double> MC_px, MC_py, MC_pz, MC_vx, MC_vy, MC_vz, MC_vt;

    outTree->Branch("beta", &beta);
    outTree->Branch("charge", &charge);
    outTree->Branch("chi2pid", &chi2pid);
    outTree->Branch("pid", &pid);
    outTree->Branch("p_x", &px);
    outTree->Branch("p_y", &py);
    outTree->Branch("p_z", &pz);
    outTree->Branch("p", &p);
    outTree->Branch("status", &status);
    outTree->Branch("v_t", &vt);
    outTree->Branch("v_x", &vx);
    outTree->Branch("v_y", &vy);
    outTree->Branch("v_z", &vz);
    outTree->Branch("theta", &theta);
    outTree->Branch("phi", &phi);

    outTree->Branch("track_charge", &track_charge);
    outTree->Branch("sector", &sector);
    outTree->Branch("NDF", &track_ndf);
    outTree->Branch("chi2", &track_chi2);

    outTree->Branch("E_PCAL", &E_PCAL);
    outTree->Branch("E_ECIN", &E_ECIN);
    outTree->Branch("E_ECOUT", &E_ECOUT);
    outTree->Branch("PCAL_U", &PCAL_U);
    outTree->Branch("PCAL_V", &PCAL_V);
    outTree->Branch("PCAL_W", &PCAL_W);

    outTree->Branch("Nphe_HTCC", &Nphe_HTCC);
    outTree->Branch("Nphe_LTCC", &Nphe_LTCC);

    outTree->Branch("DC_region1_x", &DC_region1_x);
    outTree->Branch("DC_region1_y", &DC_region1_y);
    outTree->Branch("DC_region1_z", &DC_region1_z);
    outTree->Branch("DC_region1_edge", &DC_region1_edge);

    outTree->Branch("DC_region2_x", &DC_region2_x);
    outTree->Branch("DC_region2_y", &DC_region2_y);
    outTree->Branch("DC_region2_z", &DC_region2_z);
    outTree->Branch("DC_region2_edge", &DC_region2_edge);

    outTree->Branch("DC_region3_x", &DC_region3_x);
    outTree->Branch("DC_region3_y", &DC_region3_y);
    outTree->Branch("DC_region3_z", &DC_region3_z);
    outTree->Branch("DC_region3_edge", &DC_region3_edge);

    outTree_meta->Branch("fcupgated", &fcupgated);
    outTree_meta->Branch("run_number", &run_number);
    outTree_meta->Branch("event_number", &event_number);
    outTree_meta->Branch("num_tracks", &num_tracks);

    std::cout << "Number of events: " << tree->GetEntries() << std::endl;
    int counter = 0;
    while (reader.Next()) {

        beta.clear(); chi2pid.clear(); px.clear(); py.clear(); pz.clear(); p.clear();
        charge.clear(); pid.clear(); status.clear();
        vt.clear(); vx.clear(); vy.clear(); vz.clear(); theta.clear(); phi.clear();
        track_charge.clear(); sector.clear(); track_ndf.clear(); track_chi2.clear();
        E_PCAL.clear(); E_ECIN.clear(); E_ECOUT.clear();
        PCAL_U.clear(); PCAL_V.clear(); PCAL_W.clear();
        Nphe_HTCC.clear(); Nphe_LTCC.clear();
        DC_region1_x.clear(); DC_region1_y.clear(); DC_region1_z.clear(); DC_region1_edge.clear();
        DC_region2_x.clear(); DC_region2_y.clear(); DC_region2_z.clear(); DC_region2_edge.clear();
        DC_region3_x.clear(); DC_region3_y.clear(); DC_region3_z.clear(); DC_region3_edge.clear();

        counter++;

        num_tracks = track_pindex_branch.GetSize();
        int num_parts = particle_pid_branch.GetSize();
        fcupgated = (fcupgated_branch.GetSize()>0) ? fcupgated_branch[0] : -1;
        run_number = (int) run_number_branch[0];
        event_number = (int) event_number_branch[0];
        outTree_meta->Fill();

        // if (num_tracks == 0 || num_parts == 0) continue;

        
        // Loop over each track to fill one entry per track
        for (int i = 0; i < num_tracks; i++) {

            track_charge.push_back((int) track_charge_branch[i]);
            sector.push_back((int) track_sector_branch[i]);
            track_ndf.push_back((int) track_ndf_branch[i]);
            track_chi2.push_back(track_chi2_branch[i]);

            int particle_i = (int) track_pindex_branch[i];

            if (particle_i < 0 || particle_i >= num_parts) {
                // invalid pindex
                continue;
            }

            // Particle info
            beta.push_back(particle_beta_branch[particle_i]);
            charge.push_back((int) particle_charge_branch[particle_i]);
            chi2pid.push_back(particle_chi2pid_branch[particle_i]);
            pid.push_back((int) particle_pid_branch[particle_i]);
            status.push_back((int) particle_status_branch[particle_i]);
            vt.push_back(particle_vt_branch[particle_i]);
            vx.push_back(particle_vx_branch[particle_i]);
            vy.push_back(particle_vy_branch[particle_i]);
            vz.push_back(particle_vz_branch[particle_i]);

            double px_i = particle_px_branch[particle_i];
            double py_i = particle_py_branch[particle_i];
            double pz_i = particle_pz_branch[particle_i];

            px.push_back(px_i);
            py.push_back(py_i);
            pz.push_back(pz_i);
            p.push_back(std::sqrt(px_i*px_i + py_i*py_i + pz_i*pz_i));

            theta.push_back(std::atan2(std::sqrt(px_i*px_i + py_i*py_i), pz_i));
            phi.push_back(std::atan2(py_i, px_i));


            // Reset calorimeter sums
            double E_PCAL_sum = 0;
            double E_ECIN_sum = 0;
            double E_ECOUT_sum = 0;
            double PCAL_U_sum = 0;
            double PCAL_V_sum = 0;
            double PCAL_W_sum = 0;

            // Calorimeter info
            for (std::size_t j = 0; j < calo_pindex_branch.GetSize(); j++) {
                if ((int)calo_pindex_branch[j] != particle_i) continue;

                int layer = (int)calo_layer_branch[j];
                double energy = calo_energy_branch[j];

                if (layer == 1) { // PCAL
                    E_PCAL_sum += energy;
                    PCAL_U_sum += calo_lu_branch[j];
                    PCAL_V_sum += calo_lv_branch[j];
                    PCAL_W_sum += calo_lw_branch[j];
                } else if (layer == 4) { // ECIN
                    E_ECIN_sum += energy;
                } else if (layer == 7) { // ECOUT
                    E_ECOUT_sum += energy;
                }
            }

            E_PCAL.push_back(E_PCAL_sum);
            E_ECIN.push_back(E_ECIN_sum);
            E_ECOUT.push_back(E_ECOUT_sum);
            PCAL_U.push_back(PCAL_U_sum);
            PCAL_V.push_back(PCAL_V_sum);
            PCAL_W.push_back(PCAL_W_sum);

            // Reset Cherenkov sums
            int Nphe_HTCC_sum = 0;
            int Nphe_LTCC_sum = 0;

            // Cherenkov info
            for (std::size_t j = 0; j < cherenkov_pindex_branch.GetSize(); j++) {
                if ((int)cherenkov_pindex_branch[j] != particle_i) continue;

                int detector_number = (int)cherenkov_detectorindex_branch[j];
                int num_photoelectrons = (int) cherenkov_photoelectrons_branch[j];

                if (detector_number == 15) {
                    Nphe_HTCC_sum += num_photoelectrons;
                } else if (detector_number == 16) {
                    Nphe_LTCC_sum += num_photoelectrons;
                }
            }
            
            Nphe_HTCC.push_back(Nphe_HTCC_sum);
            Nphe_LTCC.push_back(Nphe_LTCC_sum);

            // Reset DC info

            double DC_region1_x_particle = 0;
            double DC_region1_y_particle = 0;
            double DC_region1_z_particle = 0;
            double DC_region1_edge_particle = 0;

            double DC_region2_x_particle = 0;
            double DC_region2_y_particle = 0;
            double DC_region2_z_particle = 0;
            double DC_region2_edge_particle = 0;

            double DC_region3_x_particle = 0;
            double DC_region3_y_particle = 0;
            double DC_region3_z_particle = 0;
            double DC_region3_edge_particle = 0;

            // Trajectory info (DC)
            for (std::size_t j = 0; j < trajectory_pindex_branch.GetSize(); j++) {
                if ((int)trajectory_pindex_branch[j] != particle_i) continue;

                int detector_number = (int)trajectory_detectorindex_branch[j];
                if (detector_number != 6) continue; // only DC

                int layer = (int)trajectory_layer_branch[j];
                if (layer == 6) { // Region 1
                    DC_region1_x_particle += trajectory_x_branch[j];
                    DC_region1_y_particle += trajectory_y_branch[j];
                    DC_region1_z_particle += trajectory_z_branch[j];
                    DC_region1_edge_particle += trajectory_edge_branch[j];
                } else if (layer == 18) { // Region 2
                    DC_region2_x_particle += trajectory_x_branch[j];
                    DC_region2_y_particle += trajectory_y_branch[j];
                    DC_region2_z_particle += trajectory_z_branch[j];
                    DC_region2_edge_particle += trajectory_edge_branch[j];
                } else if (layer == 36) { // Region 3
                    DC_region3_x_particle += trajectory_x_branch[j];
                    DC_region3_y_particle += trajectory_y_branch[j];
                    DC_region3_z_particle += trajectory_z_branch[j];
                    DC_region3_edge_particle += trajectory_edge_branch[j];
                }
            }

            // Push per-particle DC info into event-level vectors
            DC_region1_x.push_back(DC_region1_x_particle);
            DC_region1_y.push_back(DC_region1_y_particle);
            DC_region1_z.push_back(DC_region1_z_particle);
            DC_region1_edge.push_back(DC_region1_edge_particle);

            DC_region2_x.push_back(DC_region2_x_particle);
            DC_region2_y.push_back(DC_region2_y_particle);
            DC_region2_z.push_back(DC_region2_z_particle);
            DC_region2_edge.push_back(DC_region2_edge_particle);

            DC_region3_x.push_back(DC_region3_x_particle);
            DC_region3_y.push_back(DC_region3_y_particle);
            DC_region3_z.push_back(DC_region3_z_particle);
            DC_region3_edge.push_back(DC_region3_edge_particle);


        } // end loop over tracks
        outTree->Fill();
        
        
    }
    // Saving the MC branches if desired
    if (save_MC)
    {
        reader.Restart();
        TTreeReaderArray<double> MC_pid_branch(reader, "MC::Particle::pid");
        TTreeReaderArray<double> MC_px_branch(reader, "MC::Particle::px");
        TTreeReaderArray<double> MC_py_branch(reader, "MC::Particle::py");
        TTreeReaderArray<double> MC_pz_branch(reader, "MC::Particle::pz");
        TTreeReaderArray<double> MC_vx_branch(reader, "MC::Particle::vx");
        TTreeReaderArray<double> MC_vy_branch(reader, "MC::Particle::vy");
        TTreeReaderArray<double> MC_vz_branch(reader, "MC::Particle::vz");
        TTreeReaderArray<double> MC_vt_branch(reader, "MC::Particle::vt");

        outTree_MC->Branch("MC_pid", &MC_pid);
        outTree_MC->Branch("MC_px", &MC_px);
        outTree_MC->Branch("MC_py", &MC_py);
        outTree_MC->Branch("MC_pz", &MC_pz);
        outTree_MC->Branch("MC_vx", &MC_vx);
        outTree_MC->Branch("MC_vy", &MC_vy);
        outTree_MC->Branch("MC_vz", &MC_vz);
        outTree_MC->Branch("MC_vt", &MC_vt);
        
        while(reader.Next())
        {   
            MC_pid.clear(); MC_px.clear(); MC_py.clear(); MC_pz.clear();
            MC_vx.clear(); MC_vy.clear(); MC_vz.clear(); MC_vt.clear();
            int num_MC_parts = MC_pid_branch.GetSize();
            for (int i = 0; i < num_MC_parts; i++)
            {
                MC_pid.push_back((int) MC_pid_branch[i]);
                MC_px.push_back(MC_px_branch[i]);
                MC_py.push_back(MC_py_branch[i]);
                MC_pz.push_back(MC_pz_branch[i]);
                MC_vx.push_back(MC_vx_branch[i]);
                MC_vy.push_back(MC_vy_branch[i]);
                MC_vz.push_back(MC_vz_branch[i]);
                MC_vt.push_back(MC_vt_branch[i]);
            }
            outTree_MC->Fill();
        }
        
    }
    std::cout << "Processed " << counter << " events" << std::endl;

    outFile->cd();
    outTree->Write();
    outTree_meta->Write();
    if (save_MC) outTree_MC->Write();
    outFile->Close();
    inFile->Close();
}
