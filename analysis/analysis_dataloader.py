import numpy as np

class AnalysisDataloader:
    def __init__(
        self,
        reconstructed,
        MC = None,
        shuffle = True,
        max_num_events = None,
        train_test_split = False,
        test_fraction = 0.2,
        name = "",
    ):

        self.reconstructed = reconstructed
        self.MC = MC
        self.shuffle = shuffle
        self.train_test_split = train_test_split
        self.data_name = name

        if max_num_events is not None:
            self.reconstructed = self.reconstructed[:max_num_events]
            if self.MC is not None:
                self.MC = self.MC[:max_num_events]
        self.num_events = len(self.reconstructed)
        
        if self.shuffle:
            self.shuffle_data()

        self.setup_reconstructed()
        self.setup_MC()
        
        if train_test_split:
            self.create_train_test_split(test_fraction)

        

    def setup_MC(self):
        if self.MC is None:
            self.pass_truth = None
            return
        self.MC["MC_p"] = np.sqrt(self.MC["MC_px"]**2 + self.MC["MC_py"]**2 + self.MC["MC_pz"]**2)
        self.pass_truth = (
            (self.MC["MC_p"] > 2) &
            (self.MC["MC_p"] < 8) &
            (self.MC["MC_W"] > 2) &
            (self.MC["MC_y"] < 0.8) &
            (self.MC["MC_theta_degrees"] > 5)
        )
    
    def setup_reconstructed(self):
        self.pass_reco = self.reconstructed["pass_reco"]
        self.reconstructed["vz"] = self.reconstructed["v_z"]

    def shuffle_data(self):
        perm = np.random.permutation(len(self.reconstructed))
        self.reconstructed = self.reconstructed[perm]
        if self.MC is not None:
            self.MC = self.MC[perm]
    def create_train_test_split(self, test_fraction):
        num_test = int(self.num_events * test_fraction)
        self.reconstructed_train = self.reconstructed[num_test:]
        self.reconstructed_test = self.reconstructed[:num_test]
        self.pass_reco_train = self.pass_reco[num_test:]
        self.pass_reco_test = self.pass_reco[:num_test]
        if self.MC is not None:
            self.MC_train = self.MC[num_test:]
            self.MC_test = self.MC[:num_test]
            self.pass_truth_train = self.pass_truth[num_test:]
            self.pass_truth_test = self.pass_truth[:num_test]
        else:
            self.MC_train = None
            self.MC_test = None
            self.pass_truth_train = None
            self.pass_truth_test = None
    
    def get_training_data(self):
        if self.train_test_split:
            return (
                self.reconstructed_train,
                self.MC_train,
                self.pass_reco_train,
                self.pass_truth_train
            )
        else:
            return (
                self.reconstructed,
                self.MC,
                self.pass_reco,
                self.pass_truth,
            )
    def get_testing_data(self):
        if self.train_test_split:
            return (
                self.reconstructed_test,
                self.MC_test,
                self.pass_reco_test,
                self.pass_truth_test,
            )
        else:
            return (
                self.reconstructed,
                self.MC,
                self.pass_reco,
                self.pass_truth,
            )
