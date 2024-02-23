from config.Config import Config
from app.algorithms.NegativeSelectionAlgorithm import NegativeSelection
from app.algorithms.VNegativeSelectionAlgorithm import VariableRadiusNSA


class NSAFactory:

    def __init__(self, config: Config):
        self.config = config

    def get_nsa(self, seed_nsa, radius_index, problem_dim):
        if self.config.algorithm == 'NSA':
            return NegativeSelection(detectors_nr=self.config.nsa_detectors_nr,
                                     radius=self.config.nsa_radius[radius_index],
                                     allowed_intersection=self.config.allowed_intersection,
                                     allowed_intersection_increment=self.config.allowed_intersection_increment,
                                     patience=self.config.patience,
                                     seed=seed_nsa,
                                     problem_dim=problem_dim)
        elif self.config.algorithm == 'VNSA':
            return VariableRadiusNSA(detectors_nr=self.config.nsa_detectors_nr,
                                     radius=self.config.self_radius[radius_index],
                                     alpha=self.config.alpha,
                                     non_self_covered_region_percentage=self.config.non_self_area_percentage,
                                     seed=seed_nsa,
                                     problem_dim=problem_dim)
