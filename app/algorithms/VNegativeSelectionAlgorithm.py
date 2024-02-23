import math
import os

import numpy as np
import pandas as pd
import scipy.stats
from scipy.spatial import distance
from numba import jit

from app.algorithms.NSA import NSA
from app.utils.utils import DistanceType
from tqdm import tqdm


class VariableRadiusNSA(NSA):
    def __init__(self, problem_dim, detectors_nr, radius, alpha,
                 non_self_covered_region_percentage, distance_type: DistanceType = DistanceType.euclidean, seed=None):
        """ Initialize the object

        :param problem_dim: The quantity of lymphocytes (Detectors) to be created.
        :param detectors_nr: The quantity of lymphocytes (Detectors) to be created.
        :param radius: The length of the radius of each self sample.
        :param alpha: .
        :param non_self_covered_region_percentage: Percentage of non-self region NSA should try to cover.
        :param seed: Optional seed to create randomness (used to generate detectors position)
        """
        # Datas
        super().__init__(
            seed=seed,
            distance_type=distance_type,
            detectors_nr=detectors_nr,
            r=radius,
            problem_dim=problem_dim
        )

        # Detectors
        self.alpha = alpha
        self.non_self_covered_region_percentage = non_self_covered_region_percentage

    #######################
    # Output functions
    #######################
    def save_model(self, columns, folder=""):
        array = [alc['detector_point'] for alc in self.self_tolerants_ALC]
        super()._save_model(array, columns, folder)

    def print_statistics(self, other_stats=None):
        super().print_statistics()

    def print_configuration(self, other_config=None):
        super().print_configuration(
            other_config=f'self_r={self.r}'
                         f'alpha={self.alpha}'
                         f'non_self_covered_region_percentage={self.non_self_covered_region_percentage}'
        )

    #######################
    # Private functions
    #######################
    def check_affinity(self, ALC_created, pattern):
        return super()._check_affinity(ALC_created['detector_point'], ALC_created['radius'], pattern)

    @jit
    def _train(self, train_dataset):
        print(f'\nStart training...')
        progress_bar = tqdm(desc="Number of Detectors found", total=self.detectors_nr)  # Display a progression bar

        sample_size = self.__get_sample_size()
        n_overlapped_sample = 0
        n_valid_sample = 0
        while len(self.self_tolerants_ALC) <= self.detectors_nr:
            sample, sample_radius = self.__get_sample_outside_self_radius(train_dataset)
            n_valid_sample += 1
            is_inside_other_detector = self.__check_if_sample_is_inside_other_detector(sample)

            if is_inside_other_detector:
                self.number_of_detectors_discarded += 1
                n_overlapped_sample += 1
                z = self.__get_z_score(n_overlapped_sample, sample_size)
                if z > scipy.stats.norm.ppf(self.alpha):
                    break
            elif not is_inside_other_detector:
                progress_bar.update()  # Update the progress bar adding 1 iteration to the total
                self.self_tolerants_ALC.append({'detector_point': sample, 'radius': sample_radius})

            if n_valid_sample >= sample_size:
                n_overlapped_sample = 0
                n_valid_sample = 0

        elapsed_time = progress_bar.format_interval(progress_bar.format_dict['elapsed'])
        progress_bar.close()  # Close progress bar object
        print(f'Training ended.')
        return elapsed_time

    def __get_sample_size(self):
        return 1 + max(5 / self.non_self_covered_region_percentage, 5 / (1 - self.non_self_covered_region_percentage))

    def __check_if_sample_is_inside_self_radius(self, sample, train_dataset):
        distance_self = None
        for self_sample in train_dataset:
            distance_self = distance.euclidean(sample, self_sample)
            if distance_self <= self.r:
                return True, distance_self

        return False, distance_self

    def __get_new_radius(self, actual_radius, sample_distance):
        if 0 <= sample_distance - self.r <= actual_radius:
            return sample_distance - self.r

        return actual_radius

    def __get_sample_outside_self_radius(self, train_dataset):
        sample = None
        actual_radius = float('inf')
        is_sample_inside_self_radius = True
        while is_sample_inside_self_radius:
            sample = self._create_an_ALC()
            is_sample_inside_self_radius, sample_distance = self.__check_if_sample_is_inside_self_radius(sample,
                                                                                                         train_dataset)
            actual_radius = self.__get_new_radius(actual_radius, sample_distance)

        return sample, actual_radius

    def __check_if_sample_is_inside_other_detector(self, sample):
        for detector in self.self_tolerants_ALC:
            distance_detector = distance.euclidean(sample, detector['detector_point'])
            if distance_detector <= detector['radius']:  # detector['radius']
                return True

        return False

    def __get_z_score(self, t, sample_size):
        return (t / math.sqrt(self.non_self_covered_region_percentage * (
                1 - self.non_self_covered_region_percentage) * sample_size)) - math.sqrt(
            (sample_size * self.non_self_covered_region_percentage) / (1 - self.non_self_covered_region_percentage))
