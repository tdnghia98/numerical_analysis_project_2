from methods.utils.gradient_hessian import *
from scipy.optimize import fmin


class OptimizationMethod:
    """
    The optimization Method Class is an abstract class representing a Quasi Newton method

    Initialization
    ----
    x_0: Vector (numpy array)
        Initial guess

    Attributes
    ----
    x_k: Vector (numpy array)
        Vector that make the problem minimum. When not solved, defaults to x_0
    s_k: Vector (numpy array)
        Newton direction vector. When the optimization is not initiated, this attribute is None
    nr_iteration: Number
        Number of iterations that was used to solve the problem
    """

    def __init__(self, x_0):
        self.x_0 = x_0
        self.x_k = x_0
        self.s_k = None
        self.nr_iteration = 0

    def optimize(self, problem):
        raise NotImplementedError

    def __find_alpha_exact_line_search(self, f):
        """
        Perform exact line search to determine alpha_k
        that minimizes f(x_k + alpha_k * s_k)
        (see 3.5)
        :return: alpha_k
        """

        def h(alpha):
            return f(self.x_k + alpha * self.s_k)

        alpha_0 = 0

        # disp = False prevents printing optimization result to console
        return fmin(h, alpha_0, disp=False)

    def __inexact_line_search(self):
        raise NotImplementedError

    def __find_acceptable_point(self):
        raise NotImplementedError

    def newton_optimization(self, problem, use_exact_line_search=True, display_log=True):
        """
        Solve optimization using base Newton method (see 3.3)
        :param
            problem: OptimizationProblem
        :return:
        """
        # x(*)
        self.nr_iteration = 0
        while True:
            # Update s_k
            self.__update_newton_direction(problem.function)
            alpha = 1
            if use_exact_line_search:
                alpha = self.__find_alpha_exact_line_search(problem.function)
            x_k_plus_1 = self.x_k + alpha * self.s_k

            # Break condition
            if np.array_equal(self.x_k, x_k_plus_1):
                break

            # Reassign x_k
            self.x_k = x_k_plus_1
            # Increment iteration count
            self.nr_iteration += 1

        if display_log:
            successful_message = "Optimization successful using basic newton method"
            print(successful_message)
            line_search_message = "With use of exact line search" if use_exact_line_search else None
            if line_search_message is not None:
                print(line_search_message)
            number_of_iteration_message = f"Number of iteration: {self.nr_iteration}"
            print(number_of_iteration_message)
            optimization_result_message = f"Optimization result: {self.x_k}"
            print(optimization_result_message)
            print()

        return self.x_k

    def __update_newton_direction(self, f):
        """
        Update the newton direction vector
        :param
            f:
                Objective function that needs to be optimized
        :return:
        """
        gradient_matrix = grad(f)(self.x_k)
        hessian_matrix = hessian(f)(self.x_k)
        inverse_hessian_matrix = np.linalg.inv(hessian_matrix)
        # Newton direction (see 3.3)
        self.s_k = -inverse_hessian_matrix.dot(gradient_matrix)

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
