import numpy as np
from optimization_method import OptimizationMethod 

class GoodBroyden(OptimizationMethod):
    
    """
        # Updating the Hess matrix H(k+1)
        # Using the good Broyden's method and Sherman - Morrison formula
        
        Attributes
        ----
        Hk: Previous Hessian, np.array(n,n)
        dx: delta x, xk+1-xk, np.array(n)
        dg: delta function g, gk+1-gk np.array(n)

    """
    def __update_hessian(self,Hk,dx,dg): 
        Hk += np.outer(dx-Hk@dg,Hk@dx)/(Hk@dx)@dg
        return Hk
    
    
class BadBroyden(OptimizationMethod):
    
    """
        # Updating the Hess matrix H(k+1)
        # Using the bad Broyden's method
        
        Attributes
        ----
        Hk: Previous Hessian
        dx: delta x, xk+1-xk
        dg: delta function g, gk+1-gk
        
    """
    def __update_hessian(self,Hk,dx,dg): 
        Hk += np.outer((dx - Hk@dg)/(dg@dg),dg)
        return Hk    
        
class SymmetricBroyden(OptimizationMethod):
    """
        # Updating the Hess matrix H(k+1)
        # Using the Symmetric Broyden's Condition and Sherman - Morrison Formula
        
        Attributes
        ----
        Hk: Previous Hessian
        dx: delta x, xk+1-xk
        dg: delta function g, gk+1-gk
        
    """
    def __update_hessian(self,Hk,dx,dg): 
        Hk += np.outer(dx-Hk@dg,dx-Hk@dg)/((dx-Hk@dg)@dg)
        return Hk
    