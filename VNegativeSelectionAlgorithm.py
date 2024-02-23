import math
import os

import numpy as np
import pandas as pd
import scipy.stats
from scipy.spatial import minkowski_distance, distance
from numba import jit
from utils import calculate_accuracy, DistanceType, compute_distance
from tqdm import tqdm


class VariableRadiusNSA:
    def __init__(self, train_dataset, test_dataset,
                 train_y, test_y,
                 detectors_nr, radius,
                 alpha, non_self_covered_region_percentage,
                 distance_type: DistanceType = DistanceType.euclidean,
                 seed=None):
        """ Initialize the object

        :param train_dataset: The training dataset to use. It's supposed to be a numpy ndarray.
            Due to the nature of the problem, all the samples in here should be 'self' samples.
        :param test_dataset: As above, but without the needing to have only 'self' samples.
        :param train_y: As above but containing only one column representing the ground truth (1 or 0, meaning self
            and not_self) of the training set.
        :param test_y: As above, but for the test set.
        :param detectors_nr: The number of detectors you want to generate.
        :param radius: The length of the radius of each self sample.
        :param alpha: .
        :param non_self_covered_region_percentage: Percentage of non-self region NSA should try to cover.
        :param seed: Optional seed to create randomness (used to generate detectors position)
        """
        # Datas
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.train_y = train_y
        self.test_y = test_y
        self.problem_dim = train_dataset.shape[1]  # number of columns/features in the dataset

        # Detectors
        self.detectors_nr = detectors_nr  # quantity of lymphocytes (Detectors) to be created
        self.r = radius
        self.self_tolerants_ALC = []
        self.alpha = alpha
        self.non_self_covered_region_percentage = non_self_covered_region_percentage
        self.dist_type = distance_type

        # Others
        self.seed = seed
        self.rng = np.random.RandomState(self.seed)

        # Stats variables
        self.DEBUG_training_time = 0
        self.DEBUG_total_number_of_detectors_generated = 0
        self.DEBUG_number_of_detectors_discarded = 0

    def train(self, compute_accuracy_on_train=False):
        """ Train the NSA on the given dataset.

       :param compute_accuracy_on_train: bool to set whether compute and print accuracy on training set or not.
       :return: The total elapsed time of the training.
       """
        # In this function you can, if you want, do some preprocessing on the datas. That's why the private declaration
        # don't use the 'self.train_dataset' directly but take a dataset in input instead.
        self.DEBUG_training_time = self.__train(self.train_dataset)
        if compute_accuracy_on_train:
            print(f'Train Accuracy: {self.__compute_accuracy_on_train()}')

        return self.DEBUG_training_time

    def test(self):
        """ This method will create two arrays with the tests and set new target value """
        selfs = []
        not_selfs = []

        for row in self.test_dataset:
            min_distance = float('inf')
            is_self = True
            i = 0
            while (i < len(self.self_tolerants_ALC)) and is_self:
                # Se is_self è False allora esco dal ciclo, altrimenti avanti. Questo perché qui i Detector sono già
                # addestrati, quindi tutto quello che ci cade dentro e fa restituire a check_affinity False viene
                # considerato un campione appartenente a NOT_SELF. Viene 'attaccato dal sistema immunitario' in pratica
                dist_from_detector, is_self = self.__check_affinity(self.self_tolerants_ALC[i], row)
                min_distance = min(dist_from_detector, min_distance)
                i += 1
            if is_self:
                selfs.append(np.concatenate(([min_distance, 1], row), axis=None))
            else:
                not_selfs.append(np.concatenate(([min_distance, 0], row), axis=None))

        return selfs, not_selfs

    #######################
    # Output functions
    #######################
    def self_tolerants_ALC_to_csv(self, columns, output_folder=""):
        """ Write on a CSV the detectors found. These will have a coordinate for every variable (feature) of the given
        problem (keep in mind that this means a very large file in output for high dimensional problems and with a lot
        of detectors). """
        array = [alc['detector_point'] for alc in self.self_tolerants_ALC]

        self_tolerants = pd.DataFrame(array)
        self_tolerants.columns = columns[1:]
        self_tolerants.to_csv(os.path.join(output_folder, 'self_tolerants_ALC.csv'), index=False)

    def print_stats(self):
        print(f'--- NSA Stats ---')
        print(f'Total training time: {self.DEBUG_training_time}')
        print(f'Total detectors generated: {self.DEBUG_total_number_of_detectors_generated}')
        print(f'Total detectors discarded: {self.DEBUG_number_of_detectors_discarded}')
        print('\n\n')

    def print_config(self):
        return (f'seed_nsa={self.seed}; det_nr={self.detectors_nr}; self_r={self.r}; '
              f'alpha={self.alpha}; non_self_covered_region_percentage={self.non_self_covered_region_percentage}; ')

    #######################
    # Private functions
    #######################
    def __create_an_ALC(self):
        """ Generate a randomly ALC (Antigen-presenting cell), i.e. a Detector """
        self.DEBUG_total_number_of_detectors_generated += 1
        return self.rng.random_sample(size=self.problem_dim)

    @jit
    def __train(self, train_dataset):
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
                self.DEBUG_number_of_detectors_discarded += 1
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

    def __check_affinity(self, ALC_created, pattern):
        """ Check affinity between created lymphocytes and patterns. Return True if the sample distance from the
        detector is greater or equal to its radius, False otherwise. """
        points_distance = compute_distance(ALC_created['detector_point'], pattern, self.dist_type)

        if points_distance >= ALC_created['radius']:
            # lymphocyte avoids self
            return points_distance, True
        return points_distance, False

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
            sample = self.__create_an_ALC()
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

    def __compute_accuracy_on_train(self):
        y_train_pred = []

        for row in self.train_dataset:
            is_self = True
            i = 0
            while (i < len(self.self_tolerants_ALC)) and is_self:
                _, is_self = self.__check_affinity(row, self.self_tolerants_ALC[i])
                i += 1
            if is_self:
                y_train_pred.append(1)
            else:
                y_train_pred.append(0)

        return calculate_accuracy(y_train_pred, self.train_y)