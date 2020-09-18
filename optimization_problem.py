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
        gradient: Matrix
            function's gradient - optional
    """

    def __init__(self, function, **kwargs):
        self.function = function
        self.gradient = kwargs.get('gradient')