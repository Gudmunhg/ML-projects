from SGD import gradient_descent
from logreg_gen import logistic_reg
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn import datasets
from matplotlib import pyplot as plt
digits = datasets.load_digits()

# define inputs and labels
inputs = digits.images
labels = digits.target

# flatten the image
# the value -1 means dimension is inferred from the remaining dimensions: 8x8 = 64
n_inputs = len(inputs)
classes = np.zeros((n_inputs, 10))
for i in range(n_inputs):
    classes[i, labels[i]] = 1

inputs = inputs.reshape(n_inputs, -1)

inputs_scaled = (inputs - np.mean(inputs))/(np.max(inputs)-np.mean(inputs))

input = np.ones((inputs_scaled.shape[0],inputs_scaled.shape[1]+1))
input[:,1:] = inputs_scaled

# one-liner from scikit-learn library

train_size = 0.8
test_size = 1 - train_size
X_train, X_test, Y_train, Y_test = train_test_split(
    input, classes, train_size=train_size, test_size=test_size, random_state=20)


lreg = logistic_reg(lmb=0.001)
numbers = gradient_descent(X_train, Y_train, minibatch_size=50, n_epochs = 100, max_iter = 10, t0 = 500, t1 = 5000)
beta = numbers.sgd(lreg.log_cost_L, lreg.log_cost_L_gradient)

#print(beta)
np.set_printoptions(suppress=True, precision=2)
prediction = lreg.probability(X_test, beta)
prediction = np.clip(prediction, 0.00001, 1)
error = lreg.log_cost(X_test, Y_test, beta)
print(beta.shape)
print(prediction[:10,:])
print(Y_test[:10,:])
print("cost func:", error)

exp_val = np.sum(prediction*Y_test)
tot_val = np.sum(Y_test)
print(f"Expectation value of {exp_val:.2f} correct guesses out of {tot_val}, or {100*exp_val/tot_val:.2f}% correct guesses.")


X_t = X.reshape((-1, 8, 8))
plt.axis('off')
plt.imshow(X_t[0], cmap=plt.cm.gray_r, interpolation='nearest')
plt.title("Correct: %d, Prediction: %d" % (Y_train[0], prediction[0]))
plt.show()
