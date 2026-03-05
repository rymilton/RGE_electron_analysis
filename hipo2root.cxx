#include <cstdlib>
#include <iostream>
#include <map>
#include <string>
#include <vector>
#include <cstdint>

#include "reader.h"
#include "TFile.h"
#include "TTree.h"

struct rec_particles_holder {
    std::vector<int> pid;
    std::vector<float> px, py, pz;
    std::vector<float> vx, vy, vz, vt;
    std::vector<int> charge;
    std::vector<float> beta, chi2pid;
    std::vector<short> status;

    void clear() {
        pid.clear();
        px.clear(); py.clear(); pz.clear();
        vx.clear(); vy.clear(); vz.clear(); vt.clear();
        charge.clear();
        beta.clear();
        chi2pid.clear();
        status.clear();
    }
};

struct rec_traj_holder {
    std::vector<short> pindex, index;
    std::vector<int> detector, layer;
    std::vector<float> x, y, z;
    std::vector<float> cx, cy, cz;
    std::vector<float> path, edge;

    void clear() {
        pindex.clear(); index.clear();
        detector.clear(); layer.clear();
        x.clear(); y.clear(); z.clear();
        cx.clear(); cy.clear(); cz.clear();
        path.clear(); edge.clear();
    }
};

struct rec_track_holder {
    std::vector<short> index, pindex, NDF;
    std::vector<int> q, sector;
    std::vector<float> chi2;

    void clear() {
        index.clear(); pindex.clear(); NDF.clear(); q.clear(); sector.clear(); chi2.clear();
    }
};

struct rec_calorimeter_holder {
    std::vector<short> pindex;
    std::vector<int> layer, sector;
    std::vector<float> energy, time, lu, lv, lw, du, dv, dw;

    void clear() {
        pindex.clear(); layer.clear(); sector.clear();
        energy.clear(); time.clear(); lu.clear(); lv.clear(); lw.clear();
        du.clear(); dv.clear(); dw.clear();
    }
};

struct rec_cherenkov_holder {
    std::vector<short> pindex;
    std::vector<int> detector;
    std::vector<float> nphe;

    void clear() {
        pindex.clear(); detector.clear(); nphe.clear();
    }
};

struct mc_particle_holder {
    std::vector<int> pid;
    std::vector<float> px, py, pz;
    std::vector<float> vx, vy, vz, vt;

    void clear() {
        pid.clear();
        px.clear(); py.clear(); pz.clear();
        vx.clear(); vy.clear(); vz.clear(); vt.clear();
    }
};

struct mc_event_holder {
    std::vector<short> npart, atarget, ztarget, ptarget, pbeam, ebeam, weight;
    std::vector<int> btype, targetid, processid;

    void clear() {
        npart.clear(); atarget.clear(); ztarget.clear(); ptarget.clear(); pbeam.clear();
        ebeam.clear(); weight.clear(); btype.clear(); targetid.clear(); processid.clear();
    }
};

struct run_scaler_holder {
    std::vector<float> fcupgated;

    void clear() {
        fcupgated.clear();
    }
};

struct run_config_holder {
    std::vector<int> run, event;
    
    void clear() {
        run.clear(); event.clear();
    }
};

int main(int argc, char** argv) {

    if (argc < 6) {
        std::cerr << "Usage: " << argv[0] << "input_directory input.hipo output_directory output_name save_MC(0 or 1)" << std::endl;
        return 1;
    }

    TString inputDir  = argv[1];
    TString inputFile = argv[2];
    TString outputDir = argv[3];
    TString outputFile = argv[4];
    bool saveMC = std::stoi(argv[5]) != 0;


    hipo::reader reader;
    std::cout << "Opening file: " << inputDir+inputFile << std::endl;
    reader.open(inputDir+inputFile);

    std::cout<< "Reading dictionary..." << std::endl;
    hipo::dictionary factory;
    reader.readDictionary(factory);

    hipo::bank rec_particles_bank(factory.getSchema("REC::Particle"));
    hipo::bank rec_traj_bank(factory.getSchema("REC::Traj"));
    hipo::bank rec_track_bank(factory.getSchema("REC::Track"));
    hipo::bank rec_calorimeter_bank(factory.getSchema("REC::Calorimeter"));
    hipo::bank rec_cherenkov_bank(factory.getSchema("REC::Cherenkov"));
    hipo::bank mc_particle_bank(factory.getSchema("MC::Particle"));
    hipo::bank mc_event_bank(factory.getSchema("MC::Event"));
    hipo::bank run_scaler_bank(factory.getSchema("RUN::scaler"));
    hipo::bank run_config_bank(factory.getSchema("RUN::config"));

    std::cout << "Creating output file: " << outputDir+outputFile << std::endl;
    TFile outfile(outputDir+outputFile, "RECREATE");
    TTree tree("data", "");
    std::cout << "Creating TTree and branches..." << std::endl;
    rec_particles_holder recParticles;
    rec_traj_holder recTraj;
    rec_track_holder recTrack;
    rec_calorimeter_holder recCalorimeter;
    rec_cherenkov_holder recCherenkov;
    mc_particle_holder mcParticles;
    mc_event_holder mcEvents;
    
    
    run_scaler_holder runScalers;
    run_config_holder runConfigs;

    // REC::Particles branches
    tree.Branch("REC::Particle::pid", &recParticles.pid);
    tree.Branch("REC::Particle::px", &recParticles.px);
    tree.Branch("REC::Particle::py", &recParticles.py);
    tree.Branch("REC::Particle::pz", &recParticles.pz);
    tree.Branch("REC::Particle::vx", &recParticles.vx);
    tree.Branch("REC::Particle::vy", &recParticles.vy);
    tree.Branch("REC::Particle::vz", &recParticles.vz);
    tree.Branch("REC::Particle::vt", &recParticles.vt);
    tree.Branch("REC::Particle::charge", &recParticles.charge);
    tree.Branch("REC::Particle::beta", &recParticles.beta);
    tree.Branch("REC::Particle::chi2pid", &recParticles.chi2pid);
    tree.Branch("REC::Particle::status", &recParticles.status);

    // REC::Traj branches
    tree.Branch("REC::Traj::pindex", &recTraj.pindex);
    tree.Branch("REC::Traj::index", &recTraj.index);
    tree.Branch("REC::Traj::detector", &recTraj.detector);
    tree.Branch("REC::Traj::layer", &recTraj.layer);
    tree.Branch("REC::Traj::x", &recTraj.x);
    tree.Branch("REC::Traj::y", &recTraj.y);
    tree.Branch("REC::Traj::z", &recTraj.z);
    tree.Branch("REC::Traj::cx", &recTraj.cx);
    tree.Branch("REC::Traj::cy", &recTraj.cy);
    tree.Branch("REC::Traj::cz", &recTraj.cz);
    tree.Branch("REC::Traj::path", &recTraj.path);
    tree.Branch("REC::Traj::edge", &recTraj.edge);

    // REC::Track branches
    tree.Branch("REC::Track::pindex", &recTrack.pindex);
    tree.Branch("REC::Track::index", &recTrack.index);
    tree.Branch("REC::Track::NDF", &recTrack.NDF);
    tree.Branch("REC::Track::q", &recTrack.q);
    tree.Branch("REC::Track::sector", &recTrack.sector);
    tree.Branch("REC::Track::chi2", &recTrack.chi2);

    // REC::Calorimeter branches
    tree.Branch("REC::Calorimeter::pindex", &recCalorimeter.pindex);
    tree.Branch("REC::Calorimeter::layer", &recCalorimeter.layer);
    tree.Branch("REC::Calorimeter::sector", &recCalorimeter.sector);
    tree.Branch("REC::Calorimeter::energy", &recCalorimeter.energy);
    tree.Branch("REC::Calorimeter::time", &recCalorimeter.time);
    tree.Branch("REC::Calorimeter::lu", &recCalorimeter.lu);
    tree.Branch("REC::Calorimeter::lv", &recCalorimeter.lv);
    tree.Branch("REC::Calorimeter::lw", &recCalorimeter.lw);
    tree.Branch("REC::Calorimeter::du", &recCalorimeter.du);
    tree.Branch("REC::Calorimeter::dv", &recCalorimeter.dv);
    tree.Branch("REC::Calorimeter::dw", &recCalorimeter.dw);

    // REC::Cherenkov branches
    tree.Branch("REC::Cherenkov::pindex", &recCherenkov.pindex);
    tree.Branch("REC::Cherenkov::detector", &recCherenkov.detector);
    tree.Branch("REC::Cherenkov::nphe", &recCherenkov.nphe);

    if (saveMC) {
        // MC::Particle branches
        tree.Branch("MC::Particle::pid", &mcParticles.pid);
        tree.Branch("MC::Particle::px", &mcParticles.px);
        tree.Branch("MC::Particle::py", &mcParticles.py);
        tree.Branch("MC::Particle::pz", &mcParticles.pz);
        tree.Branch("MC::Particle::vx", &mcParticles.vx);
        tree.Branch("MC::Particle::vy", &mcParticles.vy);
        tree.Branch("MC::Particle::vz", &mcParticles.vz);
        tree.Branch("MC::Particle::vt", &mcParticles.vt);

        // MC::Event branches
        tree.Branch("MC::Event::npart", &mcEvents.npart);
        tree.Branch("MC::Event::atarget", &mcEvents.atarget);
        tree.Branch("MC::Event::ztarget", &mcEvents.ztarget);
        tree.Branch("MC::Event::ptarget", &mcEvents.ptarget);
        tree.Branch("MC::Event::pbeam", &mcEvents.pbeam);
        tree.Branch("MC::Event::ebeam", &mcEvents.ebeam);
        tree.Branch("MC::Event::weight", &mcEvents.weight);
        tree.Branch("MC::Event::btype", &mcEvents.btype);
        tree.Branch("MC::Event::targetid", &mcEvents.targetid);
        tree.Branch("MC::Event::processid", &mcEvents.processid);
    }

    // RUN::scaler branches
    tree.Branch("RUN::scaler::fcupgated", &runScalers.fcupgated);

    // RUN::config branches
    tree.Branch("RUN::config::run", &runConfigs.run);
    tree.Branch("RUN::config::event", &runConfigs.event);

    hipo::event event;
    int counter = 0;
    std::cout << "Processing events..." << std::endl;
    while (reader.next() == true) {
        if (counter % 10000 == 0) {
            std::cout << "Processed " << counter << " events" << std::endl;
        }
        reader.read(event);

        recParticles.clear();
        recTraj.clear();
        recTrack.clear();
        recCalorimeter.clear();
        recCherenkov.clear();
        runScalers.clear();
        runConfigs.clear();

        event.getStructure(rec_particles_bank);
        int nrows = rec_particles_bank.getRows();
        for (int row = 0; row < nrows; row++) {
            recParticles.pid.push_back(rec_particles_bank.getInt("pid", row));
            recParticles.px.push_back(rec_particles_bank.getFloat("px", row));
            recParticles.py.push_back(rec_particles_bank.getFloat("py", row));
            recParticles.pz.push_back(rec_particles_bank.getFloat("pz", row));
            recParticles.vx.push_back(rec_particles_bank.getFloat("vx", row));
            recParticles.vy.push_back(rec_particles_bank.getFloat("vy", row));
            recParticles.vz.push_back(rec_particles_bank.getFloat("vz", row));
            recParticles.vt.push_back(rec_particles_bank.getFloat("vt", row));
            recParticles.charge.push_back(rec_particles_bank.getByte("charge", row));
            recParticles.beta.push_back(rec_particles_bank.getFloat("beta", row));
            recParticles.chi2pid.push_back(rec_particles_bank.getFloat("chi2pid", row));
            recParticles.status.push_back(rec_particles_bank.getShort("status", row));
        }

        event.getStructure(rec_traj_bank);
        nrows = rec_traj_bank.getRows();
        for (int row = 0; row < nrows; row++) {
            recTraj.pindex.push_back(rec_traj_bank.getShort("pindex", row));
            recTraj.index.push_back(rec_traj_bank.getShort("index", row));
            recTraj.detector.push_back(rec_traj_bank.getByte("detector", row));
            recTraj.layer.push_back(rec_traj_bank.getByte("layer", row));
            recTraj.x.push_back(rec_traj_bank.getFloat("x", row));
            recTraj.y.push_back(rec_traj_bank.getFloat("y", row));
            recTraj.z.push_back(rec_traj_bank.getFloat("z", row));
            recTraj.cx.push_back(rec_traj_bank.getFloat("cx", row));
            recTraj.cy.push_back(rec_traj_bank.getFloat("cy", row));
            recTraj.cz.push_back(rec_traj_bank.getFloat("cz", row));
            recTraj.path.push_back(rec_traj_bank.getFloat("path", row));
            recTraj.edge.push_back(rec_traj_bank.getFloat("edge", row));
        }
        
        event.getStructure(rec_track_bank);
        nrows = rec_track_bank.getRows();
        for (int row = 0; row < nrows; row++) {
            recTrack.index.push_back(rec_track_bank.getShort("index", row));
            recTrack.pindex.push_back(rec_track_bank.getShort("pindex", row));
            recTrack.NDF.push_back(rec_track_bank.getShort("NDF", row));
            recTrack.q.push_back(rec_track_bank.getByte("q", row));
            recTrack.sector.push_back(rec_track_bank.getByte("sector", row));
            recTrack.chi2.push_back(rec_track_bank.getFloat("chi2", row));
        }

        event.getStructure(rec_calorimeter_bank);
        nrows = rec_calorimeter_bank.getRows();
        for (int row = 0; row < nrows; row++) {
            recCalorimeter.pindex.push_back(rec_calorimeter_bank.getShort("pindex", row));
            recCalorimeter.layer.push_back(rec_calorimeter_bank.getByte("layer", row));
            recCalorimeter.sector.push_back(rec_calorimeter_bank.getByte("sector", row));
            recCalorimeter.energy.push_back(rec_calorimeter_bank.getFloat("energy", row));
            recCalorimeter.time.push_back(rec_calorimeter_bank.getFloat("time", row));
            recCalorimeter.lu.push_back(rec_calorimeter_bank.getFloat("lu", row));
            recCalorimeter.lv.push_back(rec_calorimeter_bank.getFloat("lv", row));
            recCalorimeter.lw.push_back(rec_calorimeter_bank.getFloat("lw", row));
            recCalorimeter.du.push_back(rec_calorimeter_bank.getFloat("du", row));
            recCalorimeter.dv.push_back(rec_calorimeter_bank.getFloat("dv", row));
            recCalorimeter.dw.push_back(rec_calorimeter_bank.getFloat("dw", row));
        }

        event.getStructure(rec_cherenkov_bank);
        nrows = rec_cherenkov_bank.getRows();
        for (int row = 0; row < nrows; row++) {
            recCherenkov.pindex.push_back(rec_cherenkov_bank.getShort("pindex", row));
            recCherenkov.detector.push_back(rec_cherenkov_bank.getByte("detector", row));
            recCherenkov.nphe.push_back(rec_cherenkov_bank.getFloat("nphe", row));
        }

        event.getStructure(run_scaler_bank);
        nrows = run_scaler_bank.getRows();
        for (int row = 0; row < nrows; row++) {
            runScalers.fcupgated.push_back(run_scaler_bank.getFloat("fcupgated", row));
        }

        event.getStructure(run_config_bank);
        nrows = run_config_bank.getRows();
        for (int row = 0; row < nrows; row++) {
            runConfigs.run.push_back(run_config_bank.getInt("run", row));
            runConfigs.event.push_back(run_config_bank.getInt("event", row));
        }
        
        if (saveMC) {
            mcParticles.clear();
            mcEvents.clear();
            event.getStructure(mc_particle_bank);
            nrows = mc_particle_bank.getRows();
            for (int row = 0; row < nrows; row++) {
                mcParticles.pid.push_back(mc_particle_bank.getInt("pid", row));
                mcParticles.px.push_back(mc_particle_bank.getFloat("px", row));
                mcParticles.py.push_back(mc_particle_bank.getFloat("py", row));
                mcParticles.pz.push_back(mc_particle_bank.getFloat("pz", row));
                mcParticles.vx.push_back(mc_particle_bank.getFloat("vx", row));
                mcParticles.vy.push_back(mc_particle_bank.getFloat("vy", row));
                mcParticles.vz.push_back(mc_particle_bank.getFloat("vz", row));
                mcParticles.vt.push_back(mc_particle_bank.getFloat("vt", row));
            }

            event.getStructure(mc_event_bank);
            nrows = mc_event_bank.getRows();
            for (int row = 0; row < nrows; row++) {
                mcEvents.npart.push_back(mc_event_bank.getShort("npart", row));
                mcEvents.atarget.push_back(mc_event_bank.getShort("atarget", row));
                mcEvents.ztarget.push_back(mc_event_bank.getShort("ztarget", row));
                mcEvents.ptarget.push_back(mc_event_bank.getShort("ptarget", row));
                mcEvents.pbeam.push_back(mc_event_bank.getShort("pbeam", row));
                mcEvents.ebeam.push_back(mc_event_bank.getShort("ebeam", row));
                mcEvents.weight.push_back(mc_event_bank.getShort("weight", row));
                mcEvents.btype.push_back(mc_event_bank.getInt("btype", row));
                mcEvents.targetid.push_back(mc_event_bank.getInt("targetid", row));
                mcEvents.processid.push_back(mc_event_bank.getInt("processid", row));
            }
        }

        tree.Fill();
        counter++;
    }

    tree.Write();
    outfile.Close();

    std::cout << "Processed events = " << counter << std::endl;
    return 0;
}
