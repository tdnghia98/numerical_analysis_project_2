import math
from methods.optimization_method import OptimizationMethod
from optimization_problem import *

def f(x):
    return x

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    problem = OptimizationProblem(f)
    x_0 = 4.
    optimize_method = OptimizationMethod(x_0)
    optimize_method.optimize(problem=problem)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
