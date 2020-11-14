#from SGD import gradient_descent
#from logreg_gen import logistic_reg
from log_lin_sgd import logistic_reg
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import numpy as np
from sklearn import datasets
from matplotlib import pyplot as plt
digits = datasets.load_digits()

# define inputs and labels
inputs = digits.images
labels = digits.target

# flatten the image
# the value -1 means dimension is inferred from the remaining dimensions: 8x8 =
n_inputs = len(inputs)
classes = np.zeros((n_inputs, 10))
for i in range(n_inputs):
    classes[i, labels[i]] = 1

inputs = inputs.reshape(n_inputs, -1)

inputs_scaled = (inputs - np.mean(inputs))/(np.max(inputs)-np.mean(inputs))

#input = np.ones((inputs_scaled.shape[0],inputs_scaled.shape[1]+1))
#input[:,1:] = inputs_scaled
input = inputs_scaled

# one-liner from scikit-learn library

train_size = 0.8
test_size = 1 - train_size
X_train, X_test, Y_train, Y_test = train_test_split(
    input, classes, train_size=train_size, test_size=test_size, random_state=20)


lreg = logistic_reg(lmb=0.001)

lreg.log_solver(X_train, Y_train, minibatch_size=250, n_epochs = 100, max_iter = 50, t0 = 500, t1 = 5000)
prediction_log = lreg.log_prediction(X_test)

labels_train = np.argmax(Y_train, axis=1)
clf = LogisticRegression(random_state=0).fit(X_train, labels_train)
prediction_skl = np.zeros_like(Y_train)
prediction_skl_vec = clf.predict(X_test)
for n in range(np.size(Y_test, axis=0)):
    prediction_skl[n, int(prediction_skl_vec[n])] = 1

print(prediction_skl)
column_labels = [str(x) for x in range(10)]

lreg.log_confusion(prediction_log, Y_test, column_labels, ["Digit", "Digit"], "Confusion Matrix of predicted digits, MNIST, using Logistic Regression")

lreg.log_confusion(prediction_skl, Y_test, column_labels, ["Digit", "Digit"], "Confusion Matrix of predicted digits, MNIST, using SKLearn Logistic Regression")
#numbers = gradient_descent(X_train, Y_train, minibatch_size=250, n_epochs = 100, max_iter = 50, t0 = 500, t1 = 5000)
"""beta = numbers.sgd(lreg.log_cost_L, lreg.log_cost_L_gradient)

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
#print(f"Expectation value of {exp_val:.2f} correct guesses out of {tot_val}, or {100*exp_val/tot_val:.2f}% correct guesses.")

labels_train = np.argmax(Y_train, axis=1)
print(labels_train[:10])
clf = LogisticRegression(random_state=0).fit(X_train, labels_train)
prediction_skl = clf.predict(X_test)

X_t = X_test.reshape((-1, 8, 8))
#plt.axis('off')
error_count = int(0)
error_shape = np.zeros(10, dtype=int)
error_count_skl = int(0)
error_shape_skl = np.zeros(10, dtype=int)

for i in range(len(Y_test)):
    if np.argmax(Y_test[i]) != np.argmax(prediction[i]):

        plt.imshow(X_t[i], cmap=plt.cm.gray_r, interpolation='nearest')
        plt.title("Correct: %d, Prediction: %d" % (np.argmax(Y_test[i]), np.argmax(prediction[i])))
        plt.show()

        error_count += 1
        error_shape[int(np.argmax(Y_test[i]))] += 1
    if np.argmax(Y_test[i]) != prediction_skl[i]:
        error_count_skl += 1
        error_shape_skl[int(np.argmax(Y_test[i]))] += 1


print(f"Our solution: {error_count} errors out of {int(tot_val)} guesses, {100*error_count/tot_val:.2f}% incorrect guesses.")
print(f"SKL solution: {error_count_skl} errors out of {int(tot_val)} guesses, {100*error_count_skl/tot_val:.2f}% incorrect guesses.")
print(error_shape)
print(error_shape_skl)

fig, ax = plt.subplots()
ax.bar(range(10), error_shape)
plt.title("Error Distribution")
plt.xlabel("Digit")
plt.ylabel("Number of incorrect guesses")
plt.show()
"""
