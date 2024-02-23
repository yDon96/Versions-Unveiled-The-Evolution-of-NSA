import os
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

from app.models.IAlgorithmModel import IAlgorithmModel
from app.models.IPrintable import IPrintable
from app.utils.utils import DistanceType
from app.utils.utils import calculate_accuracy, compute_distance


class NSA(IAlgorithmModel, IPrintable, ABC):

    def __init__(self, problem_dim, r, detectors_nr, seed=None, distance_type: DistanceType = DistanceType.minkowski):
        self.self_tolerants_ALC = []
        self.seed = seed
        self.rng = np.random.RandomState(seed)
        self.dist_type = distance_type
        self.detectors_nr = detectors_nr
        self.r = r
        self.problem_dim = problem_dim
        self.training_time = 0
        self.total_number_of_detectors_generated = 0
        self.number_of_detectors_discarded = 0

    def train(self, dataset, y, should_compute_accuracy=False):
        # In this function you can, if you want, do some preprocessing on the datas. That's why the private declaration
        # don't use the 'self.train_dataset' directly but take a dataset in input instead.
        self.training_time = self._train(dataset)
        if should_compute_accuracy:
            print(f'Train Accuracy: {self._get_accuracy_on(dataset, y)}')

        return self.training_time
    @abstractmethod
    def _train(self, dataset):
        pass

    def test(self, dataset):
        selfs = []
        not_selfs = []

        for row in dataset:
            min_distance = float('inf')
            is_self = True
            i = 0
            while (i < len(self.self_tolerants_ALC)) and is_self:
                # Se is_self è False allora esco dal ciclo, altrimenti avanti. Questo perché qui i Detector sono già
                # addestrati, quindi tutto quello che ci cade dentro e fa restituire a check_affinity False viene
                # considerato un campione appartenente a NOT_SELF. Viene 'attaccato dal sistema immunitario' in pratica
                dist_from_detector, is_self = self.check_affinity(self.self_tolerants_ALC[i], row)
                min_distance = min(dist_from_detector, min_distance)
                i += 1
            if is_self:
                selfs.append(np.concatenate(([min_distance, 1], row), axis=None))
            else:
                not_selfs.append(np.concatenate(([min_distance, 0], row), axis=None))

        return selfs, not_selfs

    def check_affinity(self, ALC_created, pattern):
        """ Check affinity between created lymphocytes and patterns. Return True if the sample distance from the
                detector is greater or equal to its radius, False otherwise. """
        return self._check_affinity(ALC_created, self.r, pattern)

    def _check_affinity(self, ALC_created, radius, pattern):
        points_distance = compute_distance(ALC_created, pattern, self.dist_type)

        if points_distance >= radius:
            # lymphocyte avoids self
            return points_distance, True
        return points_distance, False
    
    def _create_an_ALC(self):
        """ Generate a randomly ALC (Antigen-presenting cell), i.e. a Detector """
        self.total_number_of_detectors_generated += 1
        return self.rng.random_sample(size=self.problem_dim)

    def _get_accuracy_on(self, dataset, y):
        y_hat = []

        for row in dataset:
            is_self = True
            i = 0
            while (i < len(self.self_tolerants_ALC)) and is_self:
                _, is_self = self.check_affinity(self.self_tolerants_ALC[i], row)
                i += 1
            if is_self:
                y_hat.append(1)
            else:
                y_hat.append(0)

        return calculate_accuracy(y_hat, y)

    def _save_model(self, self_tolerants_ALC, columns, folder):
        """ Write on a CSV the detectors found. These will have a coordinate for every variable (feature) of the given
        problem (keep in mind that this means a very large file in output for high dimensional problems and with a lot
        of detectors). """
        self_tolerants = pd.DataFrame(self_tolerants_ALC)
        self_tolerants.columns = columns[1:]
        self_tolerants.to_csv(os.path.join(folder, 'self_tolerants_ALC.csv'), index=False)

    def print_statistics(self, other_stats=None):
        print(f'--- NSA Stats ---')
        print(f'Total training time: {self.training_time}')
        print(f'Total detectors generated: {self.total_number_of_detectors_generated}')
        print(f'Total detectors discarded: {self.number_of_detectors_discarded}')
        if other_stats:
            print(other_stats)
        print('\n\n')

    def print_configuration(self, other_config=None):
        print(f'--- NSA Config ---')
        print(f'seed_nsa={self.seed}')
        print(f'det_nr={self.detectors_nr}')
        if other_config:
            print(other_config)
        print('\n\n')
