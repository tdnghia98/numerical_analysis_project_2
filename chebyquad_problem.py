"""
Chebyquad Testproblem

Course material for the course FMNN25
Version for Python 3.4
Claus Führer (2016)

"""

import scipy.optimize as so
import numpy as np
from optimization_problem import OptimizationProblem
from methods.optimization_method import OptimizationMethod

def T(x, n):
    """
    Recursive evaluation of the Chebychev Polynomials of the first kind
    x evaluation point (scalar)
    n degree 
    """
    if n == 0:
        return 1.0
    if n == 1:
        return x
    return 2. * x * T(x, n - 1) - T(x, n - 2)

def U(x, n):
    """
    Recursive evaluation of the Chebychev Polynomials of the second kind
    x evaluation point (scalar)
    n degree 
    Note d/dx T(x,n)= n*U(x,n-1)  
    """
    if n == 0:
        return 1.0
    if n == 1:
        return 2. * x
    return 2. * x * U(x, n - 1) - U(x, n - 2) 
    
def chebyquad_fcn(x):
    """
    Nonlinear function: R^n -> R^n
    """    
    n = len(x)
    def exact_integral(n):
        """
        Generator object to compute the exact integral of
        the transformed Chebychev function T(2x-1,i), i=0...n
        """
        for i in range(n):
            if i % 2 == 0: 
                yield -1./(i**2 - 1.)
            else:
                yield 0.

    exint = exact_integral(n)
    
    def approx_integral(i):
        """
        Approximates the integral by taking the mean value
        of n sample points
        """
        return sum(T(2. * xj - 1., i) for xj in x) / n
    return np.array([approx_integral(i) - e for i,e in enumerate(exint)]) 


class Chebychev(OptimizationProblem):
    def __init__(self):
        ## degree determined by input of function call
        pass
    
    def f(self, x):
        """            
        norm(chebyquad_fcn)**2                
        """
        chq = chebyquad_fcn(x)
        return np.dot(chq, chq)

    def grad(self, x):
        """
        Evaluation of the gradient function of chebyquad
        """
        chq = chebyquad_fcn(x)
        UM = 4. / len(x) * np.array([[(i+1) * U(2. * xj - 1., i) 
                                 for xj in x] for i in range(len(x) - 1)])
        return np.dot(chq[1:].reshape((1, -1)), UM).reshape((-1, ))
    
if __name__ == '__main__':
    
    cheby_prob = Chebychev()
    opt = OptimizationMethod()
    
    for n in [4, 8, 11]:
        x0 = np.linspace(0, 1, n)
        
        newton_sol, _ = opt.newton_optimization(cheby_prob, x0, tol = 1e-8, display_log = False, linesearch = 'exact')
        scipy_sol = so.fmin_bfgs(cheby_prob.f, x0, cheby_prob.grad, disp = False)
        
        diff = np.linalg.norm(newton_sol - scipy_sol, 2)
        
        print('n = ', n)
        print('newton_sol:', newton_sol, 'residual:', cheby_prob.f(newton_sol))
        print('scipy_bfgs_sol', scipy_sol, 'residual:', cheby_prob.f(scipy_sol))
        print('difference in 2 norm: {}'.format(diff))
        print()
        
        ## Solution differs quite a bit for n = 8, 11, however, ours should be better due to smaller residual