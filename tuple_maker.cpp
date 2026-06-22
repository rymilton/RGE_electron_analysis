#include <TFile.h>
#include <TTree.h>
#include <TTreeReader.h>
#include <TTreeReaderArray.h>
#include <TTreeReaderValue.h>
#include <TString.h>
#include <iostream>
#include <cmath> // for std::sqrt, std::atan2
#include <vector>
#include <filesystem>
#include <future>
#include <thread>
#include <ROOT/TProcessExecutor.hxx>

/*
    My whole workflow is getting tedious with all the scripts that I have to run.
    I should add the following to this script:
    - Only keep events with a trigger electron -- done
    - Calculate the DIS quantities for each event and save them in an event tree -- done
    - Save the particles in a separated tree -- done
    - Do the same for MC -- done
    - Switch to RNTuple
    - Add multiprocessing if given a whole directory of files, but also allow for a single file to be processed
*/

std::tuple<double, double, double, double, double> calculate_DIS_quantities(double elec_px, double elec_py, double elec_pz)
{
    const double E_beam = 10.547;
    double p = std::sqrt(elec_px*elec_px + elec_py*elec_py + elec_pz*elec_pz); // electron momentum
    double E = std::sqrt(p*p + 0.000511*0.000511); // electron energy
    double nu = E_beam - E; // energy transfer

    double theta = std::atan2(std::sqrt(elec_px*elec_px + elec_py*elec_py), elec_pz); // electron scattering angle
    double Q2 = 4 * E_beam * E * std::sin(theta/2) * std::sin(theta/2);
    double x = Q2 / (2 * 0.938 * nu); // Bjorken x, using proton mass
    double y = nu / E_beam; // inelasticity
    double W2 = 0.938*0.938 + 2*0.938*nu - Q2; // invariant mass of the hadronic system

    return std::make_tuple(Q2, nu, x, y, W2);
}
void process_single_file(
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
    TTreeReaderArray<int> particle_pid_branch(reader,       "REC::Particle::pid");
    TTreeReaderArray<float> particle_vx_branch(reader,      "REC::Particle::vx");
    TTreeReaderArray<float> particle_vy_branch(reader,      "REC::Particle::vy");
    TTreeReaderArray<float> particle_vz_branch(reader,      "REC::Particle::vz");
    TTreeReaderArray<float> particle_vt_branch(reader,      "REC::Particle::vt");
    TTreeReaderArray<float> particle_px_branch(reader,      "REC::Particle::px");
    TTreeReaderArray<float> particle_py_branch(reader,      "REC::Particle::py");
    TTreeReaderArray<float> particle_pz_branch(reader,      "REC::Particle::pz");
    TTreeReaderArray<int> particle_charge_branch(reader,    "REC::Particle::charge");
    TTreeReaderArray<float> particle_beta_branch(reader,    "REC::Particle::beta");
    TTreeReaderArray<float> particle_chi2pid_branch(reader, "REC::Particle::chi2pid");
    TTreeReaderArray<short> particle_status_branch(reader,  "REC::Particle::status");

    TTreeReaderArray<short> track_pindex_branch(reader, "REC::Track::pindex");
    TTreeReaderArray<int> track_charge_branch(reader, "REC::Track::q");
    TTreeReaderArray<int> track_sector_branch(reader, "REC::Track::sector");
    TTreeReaderArray<short> track_ndf_branch(reader,    "REC::Track::NDF");
    TTreeReaderArray<float> track_chi2_branch(reader,   "REC::Track::chi2");

    TTreeReaderArray<short> calo_pindex_branch(reader, "REC::Calorimeter::pindex");
    TTreeReaderArray<int> calo_layer_branch(reader,  "REC::Calorimeter::layer");
    TTreeReaderArray<float> calo_energy_branch(reader, "REC::Calorimeter::energy");
    TTreeReaderArray<float> calo_lu_branch(reader,     "REC::Calorimeter::lu");
    TTreeReaderArray<float> calo_lv_branch(reader,     "REC::Calorimeter::lv");
    TTreeReaderArray<float> calo_lw_branch(reader,     "REC::Calorimeter::lw");

    TTreeReaderArray<short> cherenkov_pindex_branch(reader,         "REC::Cherenkov::pindex");
    TTreeReaderArray<int> cherenkov_detectorindex_branch(reader,  "REC::Cherenkov::detector");
    TTreeReaderArray<float> cherenkov_photoelectrons_branch(reader, "REC::Cherenkov::nphe");

    TTreeReaderArray<short> trajectory_pindex_branch(reader,        "REC::Traj::pindex");
    TTreeReaderArray<int> trajectory_detectorindex_branch(reader, "REC::Traj::detector");
    TTreeReaderArray<int> trajectory_layer_branch(reader,         "REC::Traj::layer");
    TTreeReaderArray<float> trajectory_x_branch(reader,             "REC::Traj::x");
    TTreeReaderArray<float> trajectory_y_branch(reader,             "REC::Traj::y");
    TTreeReaderArray<float> trajectory_z_branch(reader,             "REC::Traj::z");
    TTreeReaderArray<float> trajectory_edge_branch(reader,          "REC::Traj::edge");

    // TTreeReaderArray<float> fcupgated_branch(reader, "RUN::scaler::fcupgated");

    // TTreeReaderArray<int> run_number_branch(reader,   "RUN::config::run");
    // TTreeReaderArray<int> event_number_branch(reader, "RUN::config::event");


    std::cout << "Will save output to " << output_file << std::endl;
    TFile* outFile = TFile::Open(output_file, "RECREATE");
    TTree* outTree_reconstructed = new TTree("reconstructed", "Reconstructed particles and event quantities");
    // TTree *outTree_meta = new TTree("meta", "Event level info");
    TTree* outTree_gen = new TTree("gen", "Gen-level particles and event quantities");

    // Output branches: A vector of particles for each event
    // Kinematic and track quantities for each particle
    std::vector<double> beta, chi2pid, px, py, pz, p, vt, vx, vy, vz, theta, phi;
    std::vector<int> charge, pid, status;
    std::vector<int> track_charge, sector, track_ndf;
    std::vector<double> track_chi2;
    
    // Detector info for each particle
    std::vector<double> E_PCAL, E_ECIN, E_ECOUT;
    std::vector<double> PCAL_U, PCAL_V, PCAL_W;
    std::vector<double> PCAL_x, PCAL_y, PCAL_z, PCAL_edge;
    std::vector<int> Nphe_HTCC, Nphe_LTCC;
    std::vector<double> DC_region1_x, DC_region1_y, DC_region1_z, DC_region1_edge;
    std::vector<double> DC_region2_x, DC_region2_y, DC_region2_z, DC_region2_edge;
    std::vector<double> DC_region3_x, DC_region3_y, DC_region3_z, DC_region3_edge;

    // Event-level DIS quantities
    double Q2, nu, x, y, W;

    double fcupgated;
    int run_number, event_number, num_tracks;

    std::vector<int> gen_pid;
    std::vector<double> gen_px, gen_py, gen_pz, gen_vx, gen_vy, gen_vz, gen_vt;
    double gen_Q2, gen_nu, gen_x, gen_y, gen_W;

    outTree_reconstructed->Branch("beta", &beta);
    outTree_reconstructed->Branch("charge", &charge);
    outTree_reconstructed->Branch("chi2pid", &chi2pid);
    outTree_reconstructed->Branch("pid", &pid);
    outTree_reconstructed->Branch("p_x", &px);
    outTree_reconstructed->Branch("p_y", &py);
    outTree_reconstructed->Branch("p_z", &pz);
    outTree_reconstructed->Branch("p", &p);
    outTree_reconstructed->Branch("status", &status);
    outTree_reconstructed->Branch("v_t", &vt);
    outTree_reconstructed->Branch("v_x", &vx);
    outTree_reconstructed->Branch("v_y", &vy);
    outTree_reconstructed->Branch("v_z", &vz);
    outTree_reconstructed->Branch("theta", &theta);
    outTree_reconstructed->Branch("phi", &phi);

    outTree_reconstructed->Branch("track_charge", &track_charge);
    outTree_reconstructed->Branch("sector", &sector);
    outTree_reconstructed->Branch("NDF", &track_ndf);
    outTree_reconstructed->Branch("chi2", &track_chi2);

    outTree_reconstructed->Branch("E_PCAL", &E_PCAL);
    outTree_reconstructed->Branch("E_ECIN", &E_ECIN);
    outTree_reconstructed->Branch("E_ECOUT", &E_ECOUT);
    outTree_reconstructed->Branch("PCAL_U", &PCAL_U);
    outTree_reconstructed->Branch("PCAL_V", &PCAL_V);
    outTree_reconstructed->Branch("PCAL_W", &PCAL_W);

    outTree_reconstructed->Branch("Nphe_HTCC", &Nphe_HTCC);
    outTree_reconstructed->Branch("Nphe_LTCC", &Nphe_LTCC);

    outTree_reconstructed->Branch("DC_region1_x", &DC_region1_x);
    outTree_reconstructed->Branch("DC_region1_y", &DC_region1_y);
    outTree_reconstructed->Branch("DC_region1_z", &DC_region1_z);
    outTree_reconstructed->Branch("DC_region1_edge", &DC_region1_edge);

    outTree_reconstructed->Branch("DC_region2_x", &DC_region2_x);
    outTree_reconstructed->Branch("DC_region2_y", &DC_region2_y);
    outTree_reconstructed->Branch("DC_region2_z", &DC_region2_z);
    outTree_reconstructed->Branch("DC_region2_edge", &DC_region2_edge);

    outTree_reconstructed->Branch("DC_region3_x", &DC_region3_x);
    outTree_reconstructed->Branch("DC_region3_y", &DC_region3_y);
    outTree_reconstructed->Branch("DC_region3_z", &DC_region3_z);
    outTree_reconstructed->Branch("DC_region3_edge", &DC_region3_edge);

    outTree_reconstructed->Branch("PCAL_x", &PCAL_x);
    outTree_reconstructed->Branch("PCAL_y", &PCAL_y);
    outTree_reconstructed->Branch("PCAL_z", &PCAL_z);
    outTree_reconstructed->Branch("PCAL_edge", &PCAL_edge);

    outTree_reconstructed->Branch("Q2", &Q2);
    outTree_reconstructed->Branch("nu", &nu);
    outTree_reconstructed->Branch("x", &x);
    outTree_reconstructed->Branch("y", &y);
    outTree_reconstructed->Branch("W", &W);

    // outTree_meta->Branch("fcupgated", &fcupgated);
    // outTree_meta->Branch("run_number", &run_number);
    // outTree_meta->Branch("event_number", &event_number);
    // outTree_meta->Branch("num_tracks", &num_tracks);

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
        PCAL_x.clear(); PCAL_y.clear(); PCAL_z.clear(); PCAL_edge.clear();

        if (counter % 10000 == 0) {
            std::cout << "Processed " << counter << " events" << std::endl;
        }
        counter++;

        num_tracks = track_pindex_branch.GetSize();
        int num_parts = particle_pid_branch.GetSize();
        // fcupgated = (fcupgated_branch.GetSize()>0) ? fcupgated_branch[0] : -1;
        // run_number = (int) run_number_branch[0];
        // event_number = (int) event_number_branch[0];
        // outTree_meta->Fill();

        if (num_tracks == 0 || num_parts == 0) continue;

        // First loop over the reconstructed particles to check that there's a reconstructed trigger electron in the event. 
        // If found, calculate DIS quantities. If not, skip the event.
        bool has_trigger_electron = false;
        for (int i = 0; i < num_parts; i++) {
            int pid_i = (int) particle_pid_branch[i];
            int status_i = (int) particle_status_branch[i];
            if (pid_i == 11 && status_i<0)
            {
                has_trigger_electron = true;
                double px_i = particle_px_branch[i];
                double py_i = particle_py_branch[i];
                double pz_i = particle_pz_branch[i];
                auto [Q2_i, nu_i, x_i, y_i, W2_i] = calculate_DIS_quantities(px_i, py_i, pz_i);
                if (W2_i < 0)
                {
                    has_trigger_electron = false; // unphysical W2, likely a bad electron candidate, so skip the event
                    break;
                }

                Q2 = Q2_i;
                nu = nu_i;
                x = x_i;
                y = y_i;
                W = std::sqrt(W2_i);

                break; // no need to loop over more particles once we've found the trigger electron
            }
        }
        if (!has_trigger_electron) continue;

        
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
            bool hits_region1 = false;

            double DC_region2_x_particle = 0;
            double DC_region2_y_particle = 0;
            double DC_region2_z_particle = 0;
            double DC_region2_edge_particle = 0;
            bool hits_region2 = false;


            double DC_region3_x_particle = 0;
            double DC_region3_y_particle = 0;
            double DC_region3_z_particle = 0;
            double DC_region3_edge_particle = 0;
            bool hits_region3 = false;

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
                    hits_region1 = true;
                } else if (layer == 18) { // Region 2
                    DC_region2_x_particle += trajectory_x_branch[j];
                    DC_region2_y_particle += trajectory_y_branch[j];
                    DC_region2_z_particle += trajectory_z_branch[j];
                    DC_region2_edge_particle += trajectory_edge_branch[j];
                    hits_region2 = true;
                } else if (layer == 36) { // Region 3
                    DC_region3_x_particle += trajectory_x_branch[j];
                    DC_region3_y_particle += trajectory_y_branch[j];
                    DC_region3_z_particle += trajectory_z_branch[j];
                    DC_region3_edge_particle += trajectory_edge_branch[j];
                    hits_region3 = true;
                }
            }

            if (!hits_region1) {
                DC_region1_x_particle = -9999;
                DC_region1_y_particle = -9999;
                DC_region1_z_particle = -9999;
                DC_region1_edge_particle = -9999;
            }
            if (!hits_region2) {
                DC_region2_x_particle = -9999;
                DC_region2_y_particle = -9999;
                DC_region2_z_particle = -9999;
                DC_region2_edge_particle = -9999;
            }
            if (!hits_region3) {
                DC_region3_x_particle = -9999;
                DC_region3_y_particle = -9999;
                DC_region3_z_particle = -9999;
                DC_region3_edge_particle = -9999;
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

            // Trajectory info (PCAL)
            double PCAL_x_particle = 0;
            double PCAL_y_particle = 0;
            double PCAL_z_particle = 0;
            double PCAL_edge_particle = 0;
            bool hits_PCAL = false;

            for (std::size_t j = 0; j < trajectory_pindex_branch.GetSize(); j++) {
                if ((int)trajectory_pindex_branch[j] != particle_i) continue;

                int detector_number = (int)trajectory_detectorindex_branch[j];
                int layer_number = (int)trajectory_layer_branch[j];
                if (detector_number != 7 || layer_number != 1) continue; // only PCAL

                PCAL_x_particle += trajectory_x_branch[j];
                PCAL_y_particle += trajectory_y_branch[j];
                PCAL_z_particle += trajectory_z_branch[j];
                PCAL_edge_particle += trajectory_edge_branch[j];
                hits_PCAL = true;
            }
            if (hits_PCAL) {
                PCAL_x.push_back(PCAL_x_particle);
                PCAL_y.push_back(PCAL_y_particle);
                PCAL_z.push_back(PCAL_z_particle);
                PCAL_edge.push_back(PCAL_edge_particle);
            } else {
                PCAL_x.push_back(-9999);
                PCAL_y.push_back(-9999);
                PCAL_z.push_back(-9999);
                PCAL_edge.push_back(-9999);
            }


        } // end loop over tracks
        outTree_reconstructed->Fill();
    }
    // Saving the MC branches if desired
    if (save_MC)
    {
        reader.Restart();
        TTreeReaderArray<int> gen_pid_branch(reader, "MC::Particle::pid");
        TTreeReaderArray<float> gen_px_branch(reader, "MC::Particle::px");
        TTreeReaderArray<float> gen_py_branch(reader, "MC::Particle::py");
        TTreeReaderArray<float> gen_pz_branch(reader, "MC::Particle::pz");
        TTreeReaderArray<float> gen_vx_branch(reader, "MC::Particle::vx");
        TTreeReaderArray<float> gen_vy_branch(reader, "MC::Particle::vy");
        TTreeReaderArray<float> gen_vz_branch(reader, "MC::Particle::vz");
        TTreeReaderArray<float> gen_vt_branch(reader, "MC::Particle::vt");

        outTree_gen->Branch("gen_pid", &gen_pid);
        outTree_gen->Branch("gen_px", &gen_px);
        outTree_gen->Branch("gen_py", &gen_py);
        outTree_gen->Branch("gen_pz", &gen_pz);
        outTree_gen->Branch("gen_vx", &gen_vx);
        outTree_gen->Branch("gen_vy", &gen_vy);
        outTree_gen->Branch("gen_vz", &gen_vz);
        outTree_gen->Branch("gen_vt", &gen_vt);

        outTree_gen->Branch("gen_Q2", &gen_Q2);
        outTree_gen->Branch("gen_nu", &gen_nu);
        outTree_gen->Branch("gen_x", &gen_x);
        outTree_gen->Branch("gen_y", &gen_y);
        outTree_gen->Branch("gen_W", &gen_W);
        
        while(reader.Next())
        {   
            gen_pid.clear(); gen_px.clear(); gen_py.clear(); gen_pz.clear();
            gen_vx.clear(); gen_vy.clear(); gen_vz.clear(); gen_vt.clear();
            int num_gen_parts = gen_pid_branch.GetSize();
            bool calculated_gen_DIS = false;
            for (int i = 0; i < num_gen_parts; i++)
            {
                int gen_pid_i = (int) gen_pid_branch[i];
                // Assuming that the first electron found is the scattered electron
                if (gen_pid_i == 11 && !calculated_gen_DIS) // if it's an electron, calculate the gen-level DIS quantities for the event
                {
                    double px_i = gen_px_branch[i];
                    double py_i = gen_py_branch[i];
                    double pz_i = gen_pz_branch[i];
                    auto [gen_Q2_i, gen_nu_i, gen_x_i, gen_y_i, gen_W2_i] = calculate_DIS_quantities(px_i, py_i, pz_i);
                    gen_Q2 = gen_Q2_i;
                    gen_nu = gen_nu_i;
                    gen_x = gen_x_i;
                    gen_y = gen_y_i;
                    gen_W = std::sqrt(std::max(0.0, gen_W2_i));
                    calculated_gen_DIS = true; // only calculate DIS quantities for the first electron found in the gen particles list, which should be the trigger electron
                }
                gen_pid.push_back((int) gen_pid_branch[i]);
                gen_px.push_back(gen_px_branch[i]);
                gen_py.push_back(gen_py_branch[i]);
                gen_pz.push_back(gen_pz_branch[i]);
                gen_vx.push_back(gen_vx_branch[i]);
                gen_vy.push_back(gen_vy_branch[i]);
                gen_vz.push_back(gen_vz_branch[i]);
                gen_vt.push_back(gen_vt_branch[i]);
            }
            outTree_gen->Fill();
        }
        
    }
    
    std::cout << "Processed " << counter << " events" << std::endl;
    outFile->cd();
    outTree_reconstructed->Write();
    // outTree_meta->Write();
    if (save_MC)
    {
        outTree_gen->Write();
    }
    outFile->Close();
    inFile->Close();
}

void tuple_maker(
    TString input_path,
    TString output_dir,
    Bool_t save_MC,
    int n_workers = 4
)
{
    std::vector<TString> files;

    // Single file case
    if (input_path.EndsWith(".root"))
    {
        files.push_back(input_path);
    }
    // Directory case
    else
    {
        for (const auto& entry : std::filesystem::directory_iterator(input_path.Data()))
        {
            if (entry.path().extension() == ".root")
            {
                files.push_back(entry.path().filename().c_str());
            }
        }
    }


    std::cout 
        << "Found "
        << files.size()
        << " files"
        << std::endl;


    // Function to generate output name
    auto make_output_name = [&](TString filename)
    {
        TString run_number = filename;

        run_number.ReplaceAll(".root", "");

        Ssiz_t pos = run_number.Last('.');
        if (pos != kNPOS)
        {
            run_number = run_number(
                pos + 1,
                run_number.Length() - pos - 1
            );
        }

        TString outname;

        outname.Form(
            "%s/ntuples_%s.root",
            output_dir.Data(),
            run_number.Data()
        );

        return outname;
    };
    



    // Single file case
    if (files.size() == 1)
    {
        TString filename = files[0];

        TString dir;
        TString file;


        if (input_path.EndsWith(".root"))
        {
            std::filesystem::path p(input_path.Data());

            dir  = p.parent_path().string();
            file = p.filename().string();
        }
        else
        {
            dir = input_path;
            file = filename;
        }


        TString outname = make_output_name(file);


        process_single_file(
            dir,
            file,
            output_dir,
            outname,
            save_MC
        );

        return;
    }

    struct FileTask
    {
        TString input_dir;
        TString file;
        TString output_dir;
        TString output_file;
        Bool_t save_MC;
    };

    std::cout 
        << "Processing "
        << files.size()
        << " files using "
        << n_workers
        << " workers"
        << std::endl;



    ROOT::TProcessExecutor pool(n_workers);
    

    std::vector<FileTask> tasks;

    for (auto& file : files)
    {
        TString outname = make_output_name(file);
        tasks.push_back(
            {
                input_path,
                file,
                output_dir,
                outname,
                save_MC
            }
        );
    }


    auto out = pool.Map(
        [](FileTask task)
        {

            process_single_file(
                task.input_dir,
                task.file,
                task.output_dir,
                task.output_file,
                task.save_MC
            );

            return 0;

        },
        tasks
    );
}


    // std::cout 
    //     << "Finished processing directory"
    //     << std::endl;