import numpy as np


class AdalineCD:
    def __init__(self, eta=0.1, n_iter=100, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state
    
    def fit(self, x, y):
        rgen = np.random.RandomState(self.random_state)
        self.w = rgen.normal(loc=0.0, scale=0.01, size=x.shape[1])
        self.b = np.float_(0.0)
        self.losses = []

        for i in range(self.n_iter):
            net_input = self.net_input(x)
            output = self.activation(net_input)
            errors = (y - output)
            self.w += self.eta * 2.0 * x.T.dot(errors) / x.shape[0]
            self.b += self.eta * 2.0 * errors.mean()
            loss = (errors**2).mean()
            self.losses.append(loss)
        return self
    
    def net_input(self, x):
        return np.dot(x, self.w) + self.b
    
    def activation(self, x):
        return x
    
    def predict(self, x):
        return np.where(self.activation(self.net_input(x)) >= 0.5, 1, 0)

class AdalineSCD:
    def __init__(self, eta=0.01, n_iter=10,
                 shuffle=True, random_state=None):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state
        self.shuffle = shuffle
        self.w_initialized = False
    
    def fit(self, x, y):
        self.w_init(x.shape[1])
        self.losses = []
        for _ in range(self.n_iter):
            x, y = self.shuffle(x, y)
            losses = []
            for xi, target in zip(x, y):
                losses.append(self.update_weights(xi, target))
            avg_loss = np.mean(losses)
            self.losses.append(avg_loss)
        return self
    
    def partial_fit(self, x, y):
        if not self.w_initialized:
            self.w_init(x.shape[1])
        if y.ravel().shape[0] > 1:
            for xi, target in zip(x, y):
                self.update_weights(xi, target)
        else:
            self.update_weights(x, y)
        return self

    def shuffle(self, x, y):
        r = self.rgen.permutation(len(y))
        return x[r], y[r]
    
    def w_init(self, m):
        self.rgen = np.random.RandomState(self.random_state)
        self.w = self.rgen.normal(loc=0.0, scale=0.01, size=m)
        self.b = np.float_(0.0)
        self.w_initialized = True
