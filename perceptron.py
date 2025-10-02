import numpy as np

class Perceptron:
    def __init__(self, eta=0.01, n_iter=100, random_state=1):
        self.eta = eta
        self.n_iter= n_iter
        self.random_state = random_state
    
    def fit(self, x, y):
        rgen = np.random.RandomState(self.random_state)
        self.w = rgen.normal(loc=0.0, scale=0.01, size=x.shape[1])
        self.b = np.float_(0.0)
        self.errors = []
        
        for _ in range(self.n_iter):
            errors = 0
            for xi, target in zip(x,y):
                update = self.eta * (target - self.predict(xi))
                self.w += update * xi
                self.b += update
                errors += int(update != 0.0)
            self.errors.append(errors)
        return self
    
        
    def net_input(self, x):
        return np.dot(x, self.w) + self.b

    def predict(self, x):
        return np.where(self.net_input(x) >= 0.0, 1, 0)