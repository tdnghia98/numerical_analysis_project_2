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

    def __init__(self, roh = 0.1, sig = 0.7, tau = 0.1, kai = 9):
        self.roh = roh
        self.sig = sig
        self.tau = tau
        self.kai = kai
        if (roh > 1/2 or roh < 0 or sig <= roh or sig > 1):
            raise ValueError('Invalid choice of parameters.')
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

    def __Goldsteincondition(self, alpha_0, alpha_l, f_alpha_0, f_alpha_l, fp_alpha_l):
        LC = False
        RC = False
        if f_alpha_0 >= f_alpha_l + (1-self.roh)*(alpha_0 - alpha_l)*fp_alpha_l:
            LC = True
        if f_alpha_0 <= f_alpha_l + self.roh*(alpha_0 - alpha_l)*fp_alpha_l:
            RC = True
        return [LC, RC]

    def __Wolfecondition(self, alpha_0, alpha_l, f_alpha_0, f_alpha_l, fp_alpha_0, fp_alpha_l):
        LC = False
        RC = False
        
        if fp_alpha_0 >= self.sig*fp_alpha_l:
            LC = True
        if f_alpha_0 <= f_alpha_l + self.roh*(alpha_0 - alpha_l)*fp_alpha_l:
            RC = True
        return [LC, RC]
    

    def __extrapolation(self, alpha_0, alpha_l, fp_alpha_0, fp_alpha_l):
        delta_alpha = (alpha_0 - alpha_l)*((fp_alpha_0)/(fp_alpha_l - fp_alpha_0))
        return delta_alpha
    
    def __interpolation(self, alpha_0, alpha_l, f_alpha_0, f_alpha_l, fp_alpha_l):
        alpha_bar = ((alpha_0 - alpha_l)**2 *fp_alpha_l)/(2*(f_alpha_l - f_alpha_0 + (alpha_0 - alpha_l)*fp_alpha_l))
        return alpha_bar
    
    
    def __Block1(self, alpha_0, alpha_l, fp_alpha_0, fp_alpha_l): #extrapolation block
        delta_alpha_0 = self.__extrapolation(alpha_0, alpha_l, fp_alpha_0, fp_alpha_l)
        delta_alpha_0 = max(delta_alpha_0, self.tau*(alpha_0 - alpha_l))
        delta_alpha_0 = min(delta_alpha_0, self.kai*(alpha_0 - alpha_l))
        alpha_l = alpha_0
        alpha_0 = alpha_0 + delta_alpha_0
        return [alpha_0, alpha_l]
    
    def __Block2(self, alpha_0, alpha_l, alpha_u, f_alpha_0, f_alpha_l, fp_alpha_l): #interpolation block
        alpha_u = min(alpha_0, alpha_u)
        alpha_0_bar = self.__interpolation(alpha_0, alpha_l, f_alpha_0, f_alpha_l, fp_alpha_l)
        alpha_0_bar = max(alpha_0_bar, alpha_l + self.tau*(alpha_u - alpha_l))
        alpha_0_bar = min(alpha_0_bar, alpha_u - self.tau*(alpha_u - alpha_l))
        alpha_0 = alpha_0_bar
        return [alpha_0, alpha_u]
    

    def __find_alpha_inexact_line_search(self, f, direction, alpha_0 = None, alpha_l = 0, alpha_u = 10**(99), Goldstein = True, Wolfe = False, eps = 1e-6):
        """
        Perform inexact line search to determine alpha_k
        that minimizes f(x_k + alpha_k * s_k)
        (see 3.6 - 3.12)
        :return: alpha_k, f(x_k + alpha_k * s_k)
        """
        #which condition to be used is choosen by the user
        ## initial guess for alpha is last alpha
        ## if not available, use midpoint between alpha_u and alpha_l
        if alpha_0 is None:
            alpha_0 = (alpha_u - alpha_l)/2
        f_alpha = lambda alpha: f(self.x_k + alpha * direction)
        fp_alpha = lambda alpha: (f_alpha(alpha + 0.5 * eps)  - f_alpha(alpha - 0.5 * eps)) / eps 
    
        f_alpha_l = f_alpha(alpha_l)
        f_alpha_0 = f_alpha(alpha_0)
        
        fp_alpha_l = fp_alpha(alpha_l)
        fp_alpha_0 = fp_alpha(alpha_0)
        
        ## avert division by zero in linesearch algorithm
        if abs(fp_alpha_l - fp_alpha_0) < 1e-16:
            print('RuntimeWarning: Inexact linesearch skipped to avert division by zero')
            return [1, f_alpha(1)]
        
        if (Goldstein == False and Wolfe == False) or (Goldstein == True and Wolfe == True):
            raise NameError('Choose Goldstein or Wolfe condition.') 
        if Goldstein == True:
            condition = self.__Goldsteincondition(alpha_0, alpha_l, f_alpha_0, f_alpha_l, fp_alpha_l)
        if Wolfe == True:
            condition = self.__Wolfecondition(alpha_0, alpha_l, f_alpha_0, f_alpha_l, fp_alpha_0, fp_alpha_l)
            
        while not (condition[0] and condition[1]):                
            if condition[0] == False:
                temp = self.__Block1(alpha_0, alpha_l, fp_alpha_0, fp_alpha_l)
                alpha_0 = temp[0]
                alpha_l = temp[1]
            else:
                temp = self.__Block2(alpha_0, alpha_l, alpha_u, f_alpha_0, f_alpha_l, fp_alpha_l)
                alpha_0 = temp[0]
                alpha_u = temp[1]

            f_alpha_l = f_alpha(alpha_l)
            f_alpha_0 = f_alpha(alpha_0)
            fp_alpha_l = fp_alpha(alpha_l)
            fp_alpha_0 = fp_alpha(alpha_0)
            if Goldstein == True:
                condition = self.__Goldsteincondition(alpha_0, alpha_l, f_alpha_0, f_alpha_l, fp_alpha_l)
            if Wolfe == True:
                condition = self.__Wolfecondition(alpha_0, alpha_l, f_alpha_0, f_alpha_l, fp_alpha_0, fp_alpha_l)

        return [alpha_0, f_alpha(alpha_0)] 

    def newton_optimization(self, problem, x0, linesearch = None, display_log=True,
                            tol = 1e-8, maxiter = 1000, callback = None,
                            inexact_linesearch_wolfe = True,
                            inexact_linesearch_goldstein = False):
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
            if linesearch:
                if linesearch == 'exact':
                    alpha = self.__find_alpha_exact_line_search(problem.f, s_k)
                elif linesearch == 'inexact':
                    alpha = self.__find_alpha_inexact_line_search(problem.f, s_k, alpha_0 = alpha,
                                                                  Wolfe = inexact_linesearch_wolfe,
                                                                  Goldstein = inexact_linesearch_goldstein)[0]    
                else:
                    raise KeyError('invalid linesearch input')
            
            x_k_plus_1 = self.x_k + alpha * s_k
            
            if display_log:
                print('iteration:', nr_iterations)
                print('linesearch alpha:', alpha)
                print('update:', np.linalg.norm(alpha * s_k, 2))
                print('residual:', np.linalg.norm(problem.grad(x_k_plus_1), 2))

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
            print()
            successful_message = "Optimization successful using basic newton method"
            print(successful_message)
            line_search_message = "With use of {} line search".format(linesearch) if linesearch else "Without linesearch"

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
