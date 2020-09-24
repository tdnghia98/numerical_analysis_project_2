#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 21 11:57:02 2020

@author: Peter Meisrimel, Lund University
"""

import numpy as np
import matplotlib.pyplot as pl
import sympy as sym
from optimization_problem import OptimizationProblem
from methods.optimization_method import OptimizationMethod
pl.close('all')

class Rosenbrock(OptimizationProblem):
    def __init__(self, exact_grad = False, exact_hessian = False):
        self.x1, self.x2 = sym.symbols('x1 x2')
        self.f_sympy = 100*(self.x2 - self.x1**2)**2 + (1 - self.x1)**2
        
        self.f = lambda x: 100*(x[1] - x[0]**2)**2 + (1 - x[0])**2
        
        if exact_grad:
            d1 = self.f_sympy.diff(self.x1)
            d2 = self.f_sympy.diff(self.x2)
            grad = lambda x: np.array([d1.subs(self.x1, x[0]).subs(self.x2, x[1]),
                                       d2.subs(self.x1, x[0]).subs(self.x2, x[1])], dtype = float)
            self.grad = grad
            
        if exact_hessian:
            d11 = self.f_sympy.diff(self.x1).diff(self.x1)
            d12 = self.f_sympy.diff(self.x1).diff(self.x2)
            d22 = self.f_sympy.diff(self.x2).diff(self.x2)
            def hessian(x):
                off_diag = d12.subs(self.x1, x[0]).subs(self.x2, x[1])
                return np.array([[d11.subs(self.x1, x[0]).subs(self.x2, x[1]), off_diag],
                                  [off_diag, d22.subs(self.x1, x[0]).subs(self.x2, x[1])]], dtype = float)
            self.hessian = hessian
            
if __name__ == '__main__':
#    rosen = Rosenbrock(exact_grad = False, exact_hessian = False)
    rosen = Rosenbrock(exact_grad = True, exact_hessian = True)
    x0 = np.array([0, -0.5]) # approximately same as project picture
    opt = OptimizationMethod()
    
    tol = 1e-8
    x_opt = np.array([1., 1.])
    
    ######################
    ## contour plotting
    n = 100
    xx = np.linspace(-1, 2, n)
    yy = np.linspace(-2, 4, n)
    X, Y = np.meshgrid(xx, yy)
    Z = np.array([[rosen.f([x, y]) for x in xx] for y in yy])
    
    pl.figure()
    ## plot contours
    cont = pl.contour(X, Y, Z, levels = np.logspace(0, 3, 10), linewidths = 1)
    pl.title('Rosenbrock contour plot')
    ######################
    
    basic = {'linesearch': None}
    ex_linesearch = {'linesearch': 'exact'}
    inex_linesearch_wolfe = {'linesearch': 'inexact', 'inexact_linesearch_wolfe': True, 'inexact_linesearch_goldstein': False}
    inex_linesearch_gold = {'linesearch': 'inexact', 'inexact_linesearch_wolfe': False, 'inexact_linesearch_goldstein': True}
    
    for runmode in [basic, ex_linesearch, inex_linesearch_wolfe, inex_linesearch_gold]:
        for hessianUpdate in range(6):
            kwargs = runmode.copy()
            
            sol, nr_iters = opt.newton_optimization(rosen, x0, callback = None, tol = tol, maxiter = 100, display_log = False,
                                             hessianUpdate = hessianUpdate, **kwargs)
            
#            assert np.linalg.norm(x_opt - sol, 2) < tol, (sol, np.linalg.norm(x_opt - sol, 2), nr_iters, runmode, hessianUpdate)
            if np.linalg.norm(x_opt - sol, 2) > tol:
                print('Convergence failure for ', runmode, 'hessianUpdate ', hessianUpdate)
                print('sol: ', sol, ' difference: ', np.linalg.norm(x_opt - sol, 2))
                print('iterations', nr_iters)
                print('\n\n')
                
                
#############
# Plot with exact linesearch
## callback function for saving iterations
iters = [np.copy(x0)]
callback = lambda x: iters.append(np.copy(x))

sol, nr_iters = opt.newton_optimization(rosen, x0, callback = callback, tol = tol, maxiter = 100, display_log = False,
                                        hessianUpdate = 0, linesearch = 'exact')

pl.figure()
## plot contours
cont = pl.contour(X, Y, Z, levels = np.logspace(0, 3, 5), linewidths = 1)
pl.clabel(cont, inline=1, fontsize=10) ## adding labels
pl.plot(*zip(*iters), marker = 'o') ## plot solution path
pl.plot(*sol, marker = 'x', color = 'red', markersize = 15) ## add marker for minimum