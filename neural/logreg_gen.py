import numpy as np

class logistic_reg: #TODO: Logreg cannot deal with minibatches and or training and stuff, need to figure out how to do that with classes.
    def __init__(self, lmb=0.1):
        #self.classes = classes #The actual classifications in the training dataset, n.c matrix
        #self.predictors = predictors #The predictor set corresponding to those classifications, n.p matrix.
        #beta has dimensions p.c
        self.lmb = lmb
        #self.n = np.size(predictors, axis=0) #number of different objects to identify
        #self.p = np.size(predictors, axis=1) #number of predictors for each object
        #self.c = np.size(classes, axis=1) #number of classifications

    def expo(self, X, beta):
        print("xbet", np.max(X @ beta))
        print("exp", np.max(np.exp(X @ beta)))
        quit()
        return np.exp(X @ beta)

    def probability(self, X, beta):
        #print(X.shape)
        #print(beta.shape)
        prob = np.zeros((X.shape[0], beta.shape[1]+1))
        prob[:,:-1] = np.exp(X @ beta)/(1 + np.sum(np.exp(X @ beta), keepdims=True, axis=1))
        prob[:,-1] = 1/(1 + np.sum(np.exp(X @ beta), keepdims=False, axis=1))
        return prob

    def log_cost(self, X, y, beta):
        cost = - y * np.log(self.probability(X, beta))
        n = y.shape[0]
        return np.sum(cost)/n

    def log_cost_gradient(self, X, y, beta):
        return -X.T @ (y - self.probability(X, beta))[:,:-1]

    def log_cost_L(self, X, y, beta):
        cost = - y * np.log(self.probability(X, beta)) + self.lmb*np.sum(beta**2)
        n = y.shape[0]
        return np.sum(cost)/n

    def log_cost_L_gradient(self, X, y, beta):
        return -X.T @ (y - self.probability(X, beta))[:,:-1] + 2*self.lmb*beta
    """
    def double_gradient_logreg(self, beta):
        return
    """


    def log_cost_num(self, X, y, beta):
        return - np.log(np.prod(np.sum(y * self.probability(X, beta), axis=1)) + pow(10,-10)) #"numerically determined" negative log cost, just multiplying the probabilities with the true classifications to remove the erroneous ones, summing to make a vector, multiplying all the vectors together to get the probability of all guesses being right, then negative log.
