import math
from methods.optimization_method import OptimizationMethod
from optimization_problem import *
import numpy as np

## DEPRECATED

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    rosenbrock_vec = np.vectorize(rosenbrock)
    x_0 = np.array([1, 5])

    problem = OptimizationProblem(rosenbrock)
    optimize_method = OptimizationMethod(x_0)
    x_k = optimize_method.newton_optimization(problem=problem, use_exact_line_search=False)
    x_k = optimize_method.newton_optimization(problem=problem, use_exact_line_search=True)
# See PyCharm help at https://www.jetbrains.com/help/pycharm/
