
import numpy as np

class gradient_descent:
    def __init__(self, X, y, minibatch_size, n_epochs = 50, max_iter = 50, t0 = 5, t1 = 50):
        self.rng = np.random.default_rng()
        self.X = X #design matrix
        self.y = y #data points
        self.minibatch_size = minibatch_size #size of each minibatch

        self.classes = self.y.shape[1]
        if self.classes > 1:
            self.classes -= 1
        self.complexity = self.X.shape[1]


        self.t0 = t0
        self.t1 = t1 #controlling the learning rate
        self.n_epochs = n_epochs
        self.max_iter = max_iter

        self.data_count = np.size(y, axis=0)
        self.minibatch_count = self.data_count/minibatch_size

    def learning_schedule(self, t):
        return self.t0/(self.t1 + t)

    def sgd(self, errorfunc, gradient):

        tol = 0.0001
        sgd_error = pow(10,10)
        theta_best = np.zeros((self.complexity,self.classes))
        for iter in range(self.max_iter):

            theta = np.random.randn(self.complexity,self.classes)

            for epoch in range(self.n_epochs):
                i = 0
                shuffle = self.rng.choice(self.data_count, size=self.data_count, replace=False) #shuffling the data
                self.X = self.X[shuffle]
                self.y = self.y[shuffle]
                model_batches = [self.X[k:k + self.minibatch_size] for k in np.arange(0, self.data_count, self.minibatch_size)]
                data_batches = [self.y[k:k + self.minibatch_size] for k in np.arange(0, self.data_count, self.minibatch_size)]
                for (model_batch, data_batch) in zip(model_batches, data_batches):
                    eta = self.learning_schedule((epoch*self.minibatch_count + i)*self.minibatch_size)
                    i += 1
                    #print("error", errorfunc(self.X, self.y, theta))
                    #print(model_batch.shape)
                    #print(data_batch.shape)
                    #print(theta.shape)
                    #print("gradient", np.max(gradient(model_batch, data_batch, theta)))
                    theta -= eta * gradient(model_batch, data_batch, theta)
            new_error = errorfunc(self.X, self.y, theta)
            #print("Error =", new_error)
            if new_error < sgd_error:
                sgd_error = new_error
                theta_best = theta

        #print(new_error)
        return theta_best
