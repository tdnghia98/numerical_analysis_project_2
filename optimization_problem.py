import utils.gradient_hessian as utils_grad_hess

class OptimizationProblem:
    """
        # Optimization problem class
        ## Typical problem class

        Let f: Rn -> R the objective function
        Find x* such that f(x*) = {x € Rn | min f(x)}
    """

    def __init__(self):
        pass
        
    def f(self, x):
        return None
    
    def grad(self, x):
        return utils_grad_hess.grad(self.f)(x)
    
    def hessian(self, x):
        return utils_grad_hess.hessian(self.f)(x)