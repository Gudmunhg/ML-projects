import numpy as np

np.random.seed(2020)

n = 6
sigma = 0.2
x = np.linspace(0,1,n)
y = 3*x**2 + np.random.normal(0, sigma, n)
splits = 2

def k_fold(x, splits = 5, shuffle = False):

    indices = np.arange(x.shape[0])
    if shuffle == True:
        rng = np.random.default_rng(seed)
        rng.shuffle(indices)

    test_inds = np.array_split(indices, splits)
    train_inds = np.array_split(indices, splits)
    for i in range(splits):
        train_inds[i] = np.concatenate(np.delete(test_inds, i, 0))

    return test_inds, train_inds

test_inds, train_inds = k_fold(x,splits)
for i in range(splits):
    print(f"test_inds = {test_inds[i]}")
    print(f"train_inds = {train_inds[i]}")
