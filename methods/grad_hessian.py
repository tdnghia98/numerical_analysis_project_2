#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 19 19:35:04 2020

@author: Peter Meisrimel, Lund University
"""

import numpy as np
import pylab as pl
import scipy as sp
pl.close('all')

def unit_vec(i, n):
    ## i-th unit vector of length n
    e = np.zeros(n)
    e[i] = 1
    return e

def grad(f, eps = 1e-8):
    # by definition, central differences, likely slow
    def grad_f(x):
        n = len(x)
        return np.array([(f(x + 0.5*eps*unit_vec(i, n)) - f(x - 0.5*eps*unit_vec(i, n)))/eps for i in range(n)])
    return grad_f
    
def hessian(f, eps = 1e-4):
    # by definition, central differences, likely slow
    def hessian_f(x):
        n = len(x)
        H = np.array([[( f(x + 0.5*eps*unit_vec(i, n) + 0.5*eps*unit_vec(j, n))
                        -f(x - 0.5*eps*unit_vec(i, n) + 0.5*eps*unit_vec(j, n))
                        -f(x + 0.5*eps*unit_vec(i, n) - 0.5*eps*unit_vec(j, n))
                        +f(x - 0.5*eps*unit_vec(i, n) - 0.5*eps*unit_vec(j, n)) 
                        )/eps**2 for i in range(n)] for j in range(n)])
        return 0.5*(H + H.T)
    return hessian_f

if __name__ == '__main__':
    a = np.array(range(1,11))
    ff = lambda x: np.dot(a*x, x) # ff(x) = sum(a_i * x_i**2)
    
    grad_ff = grad(ff)
    print(np.linalg.norm(grad_ff(np.ones(10)) - 2*np.array(range(1,11)), 2))
    
    hess_ff = hessian(ff)
    print(np.linalg.norm(hess_ff(np.ones(10)) - np.diag(2*np.array(range(1,11))), 2))