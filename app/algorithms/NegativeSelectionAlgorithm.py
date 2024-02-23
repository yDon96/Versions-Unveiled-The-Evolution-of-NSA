import numpy as np
from numba import jit

from app.algorithms.NSA import NSA
from app.utils.utils import DistanceType, compute_distance, compute_intersection
from tqdm import tqdm


class NegativeSelection(NSA):
    def __init__(self, problem_dim, detectors_nr, radius, allowed_intersection=1.0,
                 allowed_intersection_increment=0.1, patience=100, distance_type: DistanceType = DistanceType.minkowski,
                 seed=None):
        """ Initialize the object

        :param detectors_nr: The number of detectors you want to generate.
        :param problem_dim: The number of detectors you want to generate.
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
        super().__init__(
            r=radius,
            detectors_nr=detectors_nr,
            seed=seed,
            distance_type=distance_type,
            problem_dim=problem_dim
        )

        # Detectors
        self.allowed_intersection = allowed_intersection
        self.allowed_intersection_increment = allowed_intersection_increment
        self.patience = patience
        self.fatigue = 0  # this is a counter for incrementing the allowed intersection after a threshold

        # Stats variables
        self.number_of_detectors_discarded_for_intersection = 0
        self.final_allowed_intersection = 0

    #######################
    # Output functions
    #######################
    def save_model(self, columns, folder=""):
        super()._save_model(self.self_tolerants_ALC, columns, folder)

    def print_statistics(self, other_stats=None):
        super().print_statistics(
            other_stats=f'Total detectors discarded due to intersection: '
                        f'{self.number_of_detectors_discarded_for_intersection}'
                        f'Total detectors discarded due to selfs sample: '
                        f'{self.number_of_detectors_discarded - self.number_of_detectors_discarded_for_intersection}'
                        f'Final allowed intersection: {self.final_allowed_intersection}'
        )

    def print_configuration(self, other_config=None):
        super().print_configuration(
            other_config=f'r={self.r}'
                         f'allowed_inter={self.allowed_intersection}'
                         f'allowed_inter_inc={self.allowed_intersection_increment}'
                         f'patience={self.patience}'
        )

    #######################
    # Private functions
    #######################
    @jit
    def _train(self, train_dataset):
        progress_bar = tqdm(desc="Number of Detectors found", total=self.detectors_nr)  # Display a progression bar

        while len(self.self_tolerants_ALC) < self.detectors_nr:
            new_alc = self._create_an_ALC()
            not_matched_selfs = True
            i = 0
            while (i < len(train_dataset)) and not_matched_selfs:
                _, not_matched_selfs = self.check_affinity(new_alc, train_dataset[i])
                i += 1
            if not_matched_selfs and self.__check_intersections(new_alc):
                progress_bar.update()  # Update the progress bar adding 1 iteration to the total
                self.self_tolerants_ALC.append(new_alc)  # Add to the found detectors the one just discovered
            else:
                self.number_of_detectors_discarded += 1

        elapsed_time = progress_bar.format_interval(progress_bar.format_dict['elapsed'])
        self.final_allowed_intersection = self.allowed_intersection
        progress_bar.close()  # Close progress bar object
        return elapsed_time

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

        self.number_of_detectors_discarded_for_intersection += 1
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
