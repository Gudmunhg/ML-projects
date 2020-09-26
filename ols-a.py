"""
Do OLS on the function FrankeFunction -
we have 2 variables so we need
1, x, x², x³, x⁴, x⁵, y, y², y³, y⁴, y⁵, xy, x²y, x³y, x⁴y, xy², x²y², x³y², xy³, x²y³, xy⁴

1+2+3+4+5+6 or in general, (n+1)*(n+2)/2 for nth degree polynomials.

Need to scale the data

"""

import numpy as np



# Make data, n meshpoints.
n = 20
x = np.arange(0, 1, 1/n)
y = np.arange(0, 1, 1/n)
x, y = np.meshgrid(x,y)

def FrankeFunction(x,y):
    term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
    term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
    term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
    term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
    return term1 + term2 + term3 + term4
z = FrankeFunction(x, y)

# The design matrix is an n^2 by m matrix, where m is the number of parameters, = (p+1)*(p+2)/2
# for polynomials of degree p, with two variables.
p = 9
m = int((p+1) * (p+2) / 2)

# First making a 3-dimensional design matrix utilizing the meshgrid. Coordinates are [y,x,degree].

Y = np.zeros((n,n,m))
k = 0

for i in range(p+1):
    for j in range(p+1-i):
        Y[:,:,k] = (y**i)*(x**j)
        k += 1

#print(Y[10,10,:])

# Converting the 3-dimensional matrix to a 2-dimensional one, starting with y_0 for the first 20 rows spanning
# x_0-x_19, then y_1 for another 20 rows, etc.

# Design Matrix
X = np.zeros((n*n,m))
# Converting the function values to a vector
Z = np.zeros(n*n)

for i in range(20):
    for j in range(20):
        X[20*i+j,:] = Y[i,j,:]
        Z[20*i+j] = z[i,j]

#print(X)
print(X.shape)

beta = np.linalg.inv(X.T @ X) @ (X.T) @ Z

Ztilde = X @ beta

print(beta)
print(Ztilde.shape)

# Converting the prediction ztilde back to a meshgrid

ztilde = np.zeros((n,n))

for i in range(20):
    ztilde[i,:] = Ztilde[20*i:20*i+20]

#print(ztilde)

def MSE(y_data,y_model):
    n = np.size(y_model)
    return np.sum((y_data-y_model)**2)/n

print(f"MSE = {MSE(z,ztilde)}")

print(z[0,0:5])
print(ztilde[0,0:5])
