from abc import ABC, abstractmethod


class IAlgorithmModel(ABC):

    @abstractmethod
    def train(self, dataset, y, should_compute_accuracy=False):
        """ Train the NSA on the given dataset.

        :param dataset: Training dataset.
        :param y: Result of each value inside the dataset.
        :param should_compute_accuracy: bool to set whether compute and print accuracy on training set or not.
        :return: The total elapsed time of the training.
        """
        pass

    @abstractmethod
    def test(self, dataset):
        """ This method will create two arrays with the tests and set new target value """
        pass

    @abstractmethod
    def _get_accuracy_on(self, dataset, y):
        pass

    @abstractmethod
    def save_model(self, columns, folder):
        pass
