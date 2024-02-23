import os

import numpy as np
import pandas as pd
from numba import jit
from utils import calculate_accuracy, DistanceType, compute_distance, compute_intersection
from tqdm import tqdm


class NSA:
    def __init__(self, train_dataset, test_dataset, train_y, test_y,
                 detectors_nr, radius, allowed_intersection=1.0,
                 allowed_intersection_increment=0.1, patience=100,
                 distance_type: DistanceType = DistanceType.minkowski,
                 seed=None):
        """ Initialize the object

        :param train_dataset: The training dataset to use. It's supposed to be a numpy ndarray.
            Due to the nature of the problem, all the samples in here should be 'self' samples.
        :param test_dataset: As above, but without the needing to have only 'self' samples.
        :param train_y: As above but containing only one column representing the ground truth (1 or 0, meaning self
            and not_self) of the training set.
        :param test_y: As above, but for the test set.
        :param detectors_nr: The number of detectors you want to generate.
        :param radius: The length of the radius of each detector.
        :param allowed_intersection: The percentage of intersection detector could have between each other (default is
            1, which is classic NSA where there is no check).
        :param allowed_intersection_increment: The value that will be added to the allowed intersection after no
            patience is left.
        :param patience: Number of detectors to throw away without find a new one before changing the allowed
            intersection.
        :param seed: Optional seed to create randomness (used to generate detectors position). Default is None, in this
            way a seed is extracted from randomness of the OS.
        """
        # Data
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.train_y = train_y
        self.test_y = test_y
        self.problem_dim = train_dataset.shape[1]  # number of columns/features in the dataset

        # Detectors
        self.detectors_nr = detectors_nr
        self.r = radius
        self.self_tolerants_ALC = []  # set of artificial lymphocytes
        self.allowed_intersection = allowed_intersection
        self.allowed_intersection_increment = allowed_intersection_increment
        self.patience = patience
        self.fatigue = 0  # this is a counter for incrementing the allowed intersection after a threshold

        # Others
        self.seed = seed
        self.rng = np.random.RandomState(self.seed)  # create random generator using the given seed to make experiments
        # repeatable
        self.dist_type = distance_type

        # Stats variables
        self.DEBUG_training_time = 0
        self.DEBUG_total_number_of_detectors_generated = 0
        self.DEBUG_number_of_detectors_discarded = 0
        self.DEBUG_number_of_detectors_discarded_for_intersection = 0
        self.DEBUG_final_allowed_intersection = 0

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
                dist_from_detector, is_self = self.__check_affinity(row, self.self_tolerants_ALC[i])
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
        self_tolerants = pd.DataFrame(self.self_tolerants_ALC)
        self_tolerants.columns = columns[1:]
        self_tolerants.to_csv(os.path.join(output_folder, 'self_tolerants_ALC.csv'), index=False)

    def print_stats(self):
        print(f'--- NSA Stats ---')
        print(f'Total training time: {self.DEBUG_training_time}')
        print(f'Total detectors generated: {self.DEBUG_total_number_of_detectors_generated}')
        print(f'Total detectors discarded: {self.DEBUG_number_of_detectors_discarded}')
        print(f'Total detectors discarded due to intersection: '
              f'{self.DEBUG_number_of_detectors_discarded_for_intersection}')
        print(f'Total detectors discarded due to selfs sample: '
              f'{self.DEBUG_number_of_detectors_discarded - self.DEBUG_number_of_detectors_discarded_for_intersection}')
        print(f'Final allowed intersection: {self.DEBUG_final_allowed_intersection}')
        print('\n\n')

    def print_config(self):
        return (f'seed_nsa={self.seed}; det_nr={self.detectors_nr}; r={self.r}; '
              f'allowed_inter={self.allowed_intersection}; allowed_inter_inc={self.allowed_intersection_increment}; '
              f'patience={self.patience}')

    #######################
    # Private functions
    #######################
    @jit
    def __train(self, train_dataset):
        progress_bar = tqdm(desc="Number of Detectors found", total=self.detectors_nr)  # Display a progression bar

        while len(self.self_tolerants_ALC) < self.detectors_nr:
            new_alc = self.__create_an_ALC()
            not_matched_selfs = True
            i = 0
            while (i < len(train_dataset)) and not_matched_selfs:
                _, not_matched_selfs = self.__check_affinity(new_alc, train_dataset[i])
                i += 1
            if not_matched_selfs and self.__check_intersections(new_alc):
                progress_bar.update()  # Update the progress bar adding 1 iteration to the total
                self.self_tolerants_ALC.append(new_alc)  # Add to the found detectors the one just discovered
            else:
                self.DEBUG_number_of_detectors_discarded += 1

        elapsed_time = progress_bar.format_interval(progress_bar.format_dict['elapsed'])
        self.DEBUG_final_allowed_intersection = self.allowed_intersection
        progress_bar.close()  # Close progress bar object
        return elapsed_time

    def __create_an_ALC(self):
        """ Generate a randomly ALC (Antigen-presenting cell), i.e. a Detector """
        self.DEBUG_total_number_of_detectors_generated += 1
        return self.rng.random_sample(size=self.problem_dim)
        # Se lo faccio qui il controllo con gli altri detector, è meglio o peggio? Conviene fare prima il check
        #  che ha meno campioni, in modo tale da scartare subito quelli inutili no? Di conseguenza sarebbe meglio se il
        #  sistema fa prima il check con i sample e poi con gli altri detector (visto che i sample sono un ordine di
        #  grandezza più piccoli solitamente).

    def __check_affinity(self, ALC_created, pattern):
        """ Check affinity between created lymphocytes and patterns. Return True if the sample distance from the
        detector is greater or equal to its radius, False otherwise. """
        points_distance = compute_distance(ALC_created, pattern, self.dist_type)

        if points_distance >= self.r:
            # lymphocyte avoids self
            return points_distance, True
        return points_distance, False

    def __check_intersections(self, new_detector) -> bool:
        """ Check if the given detector intersect with any of the others already discovered. Time complexity is O(n),
        where n is the number of detectors discovered so far. When allowed intersection is equal to 1, time complexity
        is O(1), because nothing is done except returning control to caller.

        :param new_detector: the detector to check
        :return: a boolean indicating if the detector doesn't intersect with others (True) or not (False).
        """
        if self.allowed_intersection == 1:
            return True

        i = 0
        actual_intersection = .0
        while (i < len(self.self_tolerants_ALC)) and (actual_intersection <= self.allowed_intersection):
            actual_detector = self.self_tolerants_ALC[i]
            dist = compute_distance(new_detector, actual_detector, self.dist_type)
            actual_intersection = compute_intersection(dist, self.r)
            i += 1
        if actual_intersection <= self.allowed_intersection:
            self.__reset_fatigue()
            return True

        self.DEBUG_number_of_detectors_discarded_for_intersection += 1
        self.__increasing_allowed_intersection(unit=self.allowed_intersection_increment, patience=self.patience)
        return False

    def __increasing_allowed_intersection(self, unit=0.01, patience=100):
        """ Allow an increasing intersection during the training, to avoid long waiting or infinite execution of
        the algorithm.

        :param unit: The value that will be added to the allowed intersection after no patience is left.
        :param patience: Number of detectors to throw away without find a new one before changing the allowed
            intersection.
        """
        if self.fatigue <= patience:
            self.fatigue += 1
        else:
            self.__reset_fatigue()
            new_allowed_intersection = self.allowed_intersection + unit
            print(new_allowed_intersection)
            self.allowed_intersection = min(new_allowed_intersection, 1)

    def __reset_fatigue(self):
        self.fatigue = 0

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
