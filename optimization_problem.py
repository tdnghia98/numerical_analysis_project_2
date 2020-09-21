class OptimizationProblem:
    """
        # Optimization problem class
        ## Typical problem class

        Let f: Rn -> R the objective function
        Find x* such that f(x*) = {x € Rn | min f(x)}


        Attributes
        ----
        function: Function
            objective function
    """

    def __init__(self, function):
        self.function = function
        # Check is the function is objective and raise error


    def __check_objective(self, function):
        raise NotImplementedError