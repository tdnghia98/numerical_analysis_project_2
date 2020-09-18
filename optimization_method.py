from abc import ABC, abstractmethod


class OptimizationMethod(ABC):
    """
    The optimization Method Class is an abstract class representing a Quasi Newton method

    Initialization
    ----
    x0: Number
        Initial guess
    """
    def __init__(self, x0):
        self.x0 = x0

    def optimize(self, problem):
        raise NotImplementedError

    def __exact_line_search(self):
        raise NotImplementedError

    def __inexact_line_search(self):
        raise NotImplementedError

    def __find_acceptable_point(self):
        raise NotImplementedError

    @abstractmethod
    def __update_hessian(self):
        """
        The algorithm here differs methods:
        - Broyden
        - Simple Rank 1
        - BFGS
        :return:
        """
        raise NotImplementedError
