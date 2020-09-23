#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 19 19:35:04 2020

@author: Peter Meisrimel, Lund University
"""

import numpy as np


def unit_vec(i, n):
    ## i-th unit vector of length n
    e = np.zeros(n)
    e[i] = 1
    return e


def grad(f, eps=1e-6):
    """
    Calculate the gradient of the given function (g)
    :param eps: epsilon, defaults to 1e-6
    :param f: Function which gradient should be calculated
    :param x: numpy array
        Input (x axis)
    :return: Hessian of input function
    """

    def grad_f(x):
        n = len(x)
        return np.array(
            [(f(x + 0.5 * eps * unit_vec(i, n)) - f(x - 0.5 * eps * unit_vec(i, n))) / eps for i in range(n)])

    return grad_f


def hessian(f, eps=1e-3):
    """
    Calculate the hessian of the given function (G)
    :param eps: epsilon, defaults to 1e-3
    :param f: Function which hessian should be calculated
    :param x: numpy array
        Input (x axis)
    :return: Hessian of input function
    """

    def hessian_f(x):
        n = len(x)
        H = np.array([[(f(x + 0.5 * eps * unit_vec(i, n) + 0.5 * eps * unit_vec(j, n))
                        - f(x - 0.5 * eps * unit_vec(i, n) + 0.5 * eps * unit_vec(j, n))
                        - f(x + 0.5 * eps * unit_vec(i, n) - 0.5 * eps * unit_vec(j, n))
                        + f(x - 0.5 * eps * unit_vec(i, n) - 0.5 * eps * unit_vec(j, n))
                        ) / eps ** 2 for i in range(n)] for j in range(n)])
        return 0.5 * (H + H.T)

    return hessian_f


# if __name__ == '__main__':
#     a = np.array(range(1, 11))
#     ff = lambda x: np.dot(a * x, x)  # ff(x) = sum(a_i * x_i**2)
#
#     grad_ff = grad(ff)
#
#     hess_ff = hessian(ff)
#
#     x = np.ones(10)
#
#     gradient_matrix = grad_ff(x)
#     hessian_matrix = hess_ff(x)
#
#     print(np.linalg.norm(gradient_matrix - 2 * np.array(range(1, 11)), 2))
#     print(np.linalg.norm(hessian_matrix - np.diag(2 * np.array(range(1, 11))), 2))
