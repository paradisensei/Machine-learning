from scipy import optimize


class Trainer(object):
    def __init__(self, N):
        # Make Local reference to network:
        self.N = N

    def callback_f(self, params):
        self.N.set_params(params)
        self.J.append(self.N.cost_function(self.X, self.y))

    def cost_function_wrapper(self, params, X, y):
        self.N.set_params(params)
        cost = self.N.cost_function(X, y)
        grad = self.N.compute_gradients(X, y)
        return cost, grad

    def train(self, X, y):
        # Make an internal variable for the callback function:
        self.X = X
        self.y = y

        # Make empty list to store costs:
        self.J = []

        params0 = self.N.get_params()

        options = {'maxiter': 3000, 'disp': True}
        _res = optimize.minimize(self.cost_function_wrapper, params0, jac=True,
                                 args=(X, y), options=options, callback=self.callback_f)

        self.N.set_params(_res.x)
