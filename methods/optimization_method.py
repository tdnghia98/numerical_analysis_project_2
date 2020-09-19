import autograd

class OptimizationMethod():
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
        return self.__newton_optimization(problem)

    def __exact_line_search(self):
        raise NotImplementedError

    def __inexact_line_search(self):
        raise NotImplementedError

    def __find_acceptable_point(self):
        raise NotImplementedError

    def __newton_optimization(self, problem):
        """
        Solve optimization using base Newton method (see 3.3)
        :param
            problem: OptimizationProblem
        :return:
        """
        # x(*)
        x_k = self.x0
        x_k_plus_1 = self.__newton_iteration(x_k, problem.function)
        while x_k != x_k_plus_1:
            x_k = x_k_plus_1
            x_k_plus_1 = self.__newton_iteration(x_k, problem.function)
        return x_k

    def __newton_iteration(self, f, x_k):
        """
        Incremental calculation of Basic method: Newton Iteration (see 3.3)
        :param
            x_k:
                Local minimizer, when doing full newton iteration, the x_0 should be supplied
            f:
                Objective function that needs to be optimized
        :return:
        """
        x_k_plus_1 = x_k - 1/self.__hessian(f, x_k)(x_k) * self.__gradient(f, x_k)(x_k)
        return x_k_plus_1

    def __hessian(self, f, x):
        """
        Calculate the hessian of the given function (G)
        :param f: Function which hessian should be calculated
        :param x: Double
            Input (x axis)
        :return: Hessian of input function
        """
        raise NotImplementedError

    def __gradient(self, f, x):
        """
        Calculate the gradient of the given function (g)
        :param f: Function which gradient should be calculated
        :param x: Double
            Input (x axis)
        :return: Hessian of input function
        """
        eps = 1e-8
        return (f(x + eps) - f(x))/eps


    def __update_hessian(self):
        """
        Update H(k+1)
        The algorithm here differs methods:
        - Broyden
        - Simple Rank 1
        - BFGS
        :return:
        """
        raise NotImplementedError