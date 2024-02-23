from abc import ABC, abstractmethod


class IPrintable(ABC):

    @abstractmethod
    def print_statistics(self, other_stats):
        """ This method will print model's statistics values"""
        pass

    @abstractmethod
    def print_configuration(self, other_config):
        """ This method will print all model's configuration params """
        pass