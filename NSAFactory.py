from Config import Config
from NegativeSelectionAlgorithm import NSA
from VNegativeSelectionAlgorithm import VariableRadiusNSA


class NSAFactory:

    def __init__(self, config: Config):
        self.config = config

    def get_nsa(self, seed_nsa, radius_index, X_train, X_test, y_train, y_test):
        if self.config.algorithm == 'NSA':
            return NSA(train_dataset=X_train, test_dataset=X_test, train_y=y_train, test_y=y_test,
                       detectors_nr=self.config.nsa_detectors_nr, radius=self.config.nsa_radius[radius_index],
                       allowed_intersection=self.config.allowed_intersection,
                       allowed_intersection_increment=self.config.allowed_intersection_increment,
                       patience=self.config.patience,
                       seed=seed_nsa)
        elif self.config.algorithm == 'VNSA':
            return VariableRadiusNSA(train_dataset=X_train, test_dataset=X_test, train_y=y_train, test_y=y_test,
                                     detectors_nr=self.config.nsa_detectors_nr,
                                     radius=self.config.self_radius[radius_index],
                                     alpha=self.config.alpha,
                                     non_self_covered_region_percentage=self.config.non_self_area_percentage,
                                     seed=seed_nsa)
