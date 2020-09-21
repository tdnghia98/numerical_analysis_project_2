from scipy.optimize import fmin
import numpy as np


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

    def __init__(self):
        # possibly track things, e.g., hessian evaluations
        pass

    # obsolete?
    def optimize(self, problem):
        raise NotImplementedError

    def __find_alpha_exact_line_search(self, f, direction):
        """
        Perform exact line search to determine alpha_k
        that minimizes f(x_k + alpha_k * s_k)
        (see 3.5)
        :return: alpha_k
        """

        def h(alpha):
            return f(self.x_k + alpha * direction)

        alpha_0 = np.array([0])

        # disp = False prevents printing optimization result to console
        return fmin(h, alpha_0, disp=False)

    def __inexact_line_search(self):
        raise NotImplementedError

    def __find_acceptable_point(self):
        raise NotImplementedError

    def newton_optimization(self, problem, x0, use_exact_line_search=True, display_log=True,
                            tol = 1e-8, maxiter = 100, callback = None):
        """
        Solve optimization using base Newton method (see 3.3)
        :param
            problem: OptimizationProblem
        :param 
            x0: Initial guess
        :param 
            tol: Tolerance for termination criterion
        :return:
        """
        # Initialize iteration
        self.x_k = np.copy(x0)
        
        # x(*)
        for nr_iterations in range(1, maxiter + 1):
            # Update s_k
            s_k = self.__get_newton_direction(problem)
            alpha = 1
            if use_exact_line_search:
                alpha = self.__find_alpha_exact_line_search(problem.f, s_k)
            x_k_plus_1 = self.x_k + alpha * s_k

            # Break condition, update smaller than a prescribed tolerance
            if np.linalg.norm(alpha * s_k, 2) < tol:
                break

            # Reassign x_k
            self.x_k = x_k_plus_1
            # callback function, for e.g. tracking the iteration
            if callback:
                callback(self.x_k)
            # update hessian
            self.__update_hessian()
            

        if display_log:
            successful_message = "Optimization successful using basic newton method"
            print(successful_message)
            line_search_message = "With use of exact line search" if use_exact_line_search else None
            if line_search_message is not None:
                print(line_search_message)
            number_of_iteration_message = f"Number of iteration: {nr_iterations}"
            print(number_of_iteration_message)
            optimization_result_message = f"Optimum at : {self.x_k} with f(x_opt) = {problem.f(self.x_k)}"
            print(optimization_result_message)
            print()

        return self.x_k, nr_iterations

    def __get_newton_direction(self, problem):
        """
        Get the newton direction vector
        :param
            f:
                Objective function that needs to be optimized
        :return:
            New newton direction
        """
        gradient_vector = problem.grad(self.x_k)
        hessian_matrix = problem.hessian(self.x_k)
        # Newton direction (see 3.3)
        return np.linalg.solve(hessian_matrix, -gradient_vector)

    # TODO: possibly needs some input parameters
    def __update_hessian(self):
        """
        Update H(k+1)
        The algorithm here differs methods:
        - Broyden
        - Simple Rank 1
        - BFGS
        :return:
        """
        pass
