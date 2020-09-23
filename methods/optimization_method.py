from methods.utils.gradient_hessian import *
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

    "The input for HessianUpdate is a switch which determines which option we use for updating the hessian"
    def newton_optimization(self, problem, use_exact_line_search=True, hessianUpdate = 0, display_log=True):
        """
        Solve optimization using base Newton method (see 3.3)
        :param
            problem: OptimizationProblem
        :return:
        """
        # x(*)
        self.nr_iteration = 1
        dirGen = self.__update_newton_direction(problem.function,hessianUpdate)
        while True:
            # Update s_k
            self.s_k = dirGen.__next__()
            alpha = 1
            if use_exact_line_search:
                alpha = self.__find_alpha_exact_line_search(problem.function)
            x_k_plus_1 = self.x_k + alpha * self.s_k
           
            # Break condition
            if np.sqrt(np.sum((self.x_k - x_k_plus_1)**2))<1e-10:
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

    def __update_newton_direction(self, f, hessianUpdate):
        """
        Update the newton direction vector
        :param
            f:
                Objective function that needs to be optimized
        :return:
        """
        hessGen = self.__update_hessian(f, hessianUpdate)
            
        while True:
            gradient_matrix = grad(f)(self.x_k)
            hessian_matrix = hessGen.__next__()
            
            if hessianUpdate == 0 or hessianUpdate == 5:
            # Newton direction (see 3.3)
                yield -np.linalg.solve(hessian_matrix, gradient_matrix)
            else:
                yield -np.dot(hessian_matrix,gradient_matrix)
                
    def __update_hessian(self, f, hessianUpdate):
        """
        Update H(k+1)
        The algorithm here differs methods:
        - Broyden
        - Simple Rank 1
        - BFGS
        :return:
        """
        if hessianUpdate == 0:
            while True:
                yield hessian(f)(self.x_k)
        elif hessianUpdate == 1:
            DFPgen = self.DFP(f)
            while True:
                H = DFPgen.__next__()
                yield H
        elif hessianUpdate == 2:
            GBgen = self.GoodBroyden(f)
            while True:
                H = GBgen.__next__()
                yield H
        elif hessianUpdate == 3:
            BBgen = self.BadBroyden(f)
            while True:
                H = BBgen.__next__()
                yield H
        elif hessianUpdate == 4:
            SBgen = self.SymmetricBroyden(f)
            while True:
                H = SBgen.__next__()
                yield H
        elif hessianUpdate == 5:
            BFGSgen = self.BFGS(f)
            while True:
                B = BFGSgen.__next__()
                yield B
        

    
    def DFP(self, f):
        "If its the first iteration we take the identity matrix as the Hessian"
        n = len(self.x_0)
        #H = np.eye(n)
        ##########"For now Hzero will be the actual hessian just to get the function to work"
        H = np.eye(n)
        grad_old = grad(f)(self.x_k)
        "In this while loop we generate the rest of the Hessians"
        while True:
            "We yield at the start of the loop since the first Hessian is created prior to the loop"
            yield H
            "Here get calculate the necessary gradient"
            grad_new = grad(f)(self.x_k)
            "Then we get y_k, that is, the change in gradient"
            y_k = grad_new - grad_old
            "we first get the first term; "
            numer1 = H.dot(y_k)
            tmpfactor = np.matmul(y_k.T, H)
            numer1 = np.matmul(numer1, tmpfactor)
            denom1 = np.matmul(tmpfactor, y_k)
            term1 = numer1/denom1
            "then we get the second term; "
            numer2 = np.outer(self.s_k, self.s_k)
            denom2 = np.matmul(self.s_k,     y_k)
            term2 = numer2/denom2
            "Lastly we set up the new Hessian"
            H = H - term1 - term2
            grad_old = grad_new

    def BFGS(self, f):
        "If its the first iteration we take the identity matrix as the Hessian"
        n = len(self.x_0)
        #H = np.eye(n)
        ###########"For now Hzero will be the actual hessian just to get the function to work"
        #B = hessian(f)(self.x_k)
        B = np.eye(n)
        grad_old = grad(f)(self.x_k)
        "In this while loop we generate the rest of the Hessians"
        while True:
            "We yield at the start of the loop since the first Hessian is created prior to the loop"
            yield B
            "Here get calculate the necessary gradient"
            grad_new = grad(f)(self.x_k)
            "Then we get y_k, that is, the change in gradient"
            y_k = grad_new - grad_old          
            
            
            
            "we first get the first term; "
            numer1 = B.dot(self.s_k)
            tmpfactor = np.matmul(self.s_k.T, B)
            numer1 = np.matmul(numer1, tmpfactor)
            denom1 = np.matmul(tmpfactor, y_k)
            term1 = numer1/denom1
            "then we get the second term; "
            numer2 = np.outer(y_k,  y_k)
            denom2 = np.matmul(y_k, self.s_k)
            term2 = numer2/denom2
            "Lastly we set up the new Hessian"
            B = B - term1 - term2
            grad_old = grad_new

    def GoodBroyden(self,f):
        "GoodBroyden yields the inverse hessian!!"
        n = len(self.x_0)
        "If its the first iteration we take the identity matrix as the inverse hessian"
        H = np.eye(n)
        grad_old = grad(f)(self.x_k)
        x_old = self.x_k
        "In this while loop we generate the rest of the inverse Hessians"
        while True:
            "We yield at the start of the loop since the first Hessian is created prior to the loop"
            yield H
            "Here get calculate the necessary gradient"
            grad_new = grad(f)(self.x_k)
            "Then we get dg, that is, the change in gradient"
            dg = grad_new - grad_old
            "Then we get dx, that is, the change in newton direction"
            dx = self.x_k - x_old
            if dx@dx != 0:
                x_old = self.x_k
                H += np.outer(dx-H@dg,H@dx)/(H@dx)@dg

    def BadBroyden(self,f):
        "BadBroyden yields the inverse hessian!!"
        n = len(self.x_0)
        "If its the first iteration we take the identity matrix as the inverse hessian"
        H = np.eye(n)
        grad_old = grad(f)(self.x_k)
        x_old = self.x_k
        "In this while loop we generate the rest of the inverse Hessians"
        while True:
            "We yield at the start of the loop since the first Hessian is created prior to the loop"
            yield H
            "Here get calculate the necessary gradient"
            grad_new = grad(f)(self.x_k)
            "Then we get dg, that is, the change in gradient"
            dg = grad_new - grad_old
            "Then we get dx, that is, the change in newton direction"
            dx = self.x_k - x_old
            if dx@dx != 0:
                x_old = self.x_k
                H += np.outer((dx - H@dg)/(dg@dg),dg)
                
    def SymmetricBroyden(self,f):
        "SymmetricBroyden yields the inverse hessian!!"
        n = len(self.x_0)
        "If its the first iteration we take the identity matrix as the inverse hessian"
        H = np.eye(n)
        grad_old = grad(f)(self.x_k)
        x_old = self.x_k
        "In this while loop we generate the rest of the inverse Hessians"
        while True:
            "We yield at the start of the loop since the first Hessian is created prior to the loop"
            yield H
            "Here get calculate the necessary gradient"
            grad_new = grad(f)(self.x_k)
            "Then we get dg, that is, the change in gradient"
            dg = grad_new - grad_old
            "Then we get dx, that is, the change in newton direction"
            dx = self.x_k - x_old
            if dx@dx != 0:
                x_old = self.x_k
                H += np.outer(dx-H@dg,dx-H@dg)/((dx-H@dg)@dg)







