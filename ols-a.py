"""
Do OLS on the function FrankeFunction -
we have 2 variables so we need
1, x, x², x³, x⁴, x⁵, y, y², y³, y⁴, y⁵, xy, x²y, x³y, x⁴y, xy², x²y², x³y², xy³, x²y³, xy⁴

1+2+3+4+5+6 or in general, (n+1)*(n+2)/2 for nth degree polynomials.

Need to scale the data

"""

import numpy as np
from sklearn.model_selection import train_test_split


# Make data, n meshpoints.
N = 20
x = np.arange(0, 1, 1/N)
y = np.arange(0, 1, 1/N)
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
def FrankeLinReg(p):

    x_train, x_test, y_train, y_test, z_train, z_test = train_test_split(x, y, z, test_size=0.2)

    X, Z = Mesh2Matr(x_train,y_train,z_train,p)

    print(X.shape)

    beta = np.linalg.inv(X.T @ X) @ (X.T) @ Z

    Ztilde = X @ beta

    print(beta)
    print(Ztilde.shape)

    # Converting the prediction ztilde back to a meshgrid

    n_train, m_train = x_train.shape[0],x_train.shape[1]

    ztilde = Vec2Mesh(Ztilde,n_train, m_train)

    #print(ztilde)

    return ztilde

def Vec2Mesh(Z, n, m):
    """
    Takes a vector Z and converts it into an nxm matrix, row by row.
    """
    z = np.zeros((n,m))

    for i in range(n):
        z[i,:] = Z[m * i : m * i + m]
    return z

def Mesh2Matr(x,y,z,p):
    """
    Converts a meshgrid x, y, and a dataset z into a design matrix X
    of polynomials in x, y up to degree p, and a vector Z with the
    dataset z arranged row by row.
    """
    n, m = x.shape[0], x.shape[1]

    # Number of predictors
    l = int((p+1) * (p+2) / 2)

    # First making a 3-dimensional design matrix utilizing the meshgrid. Coordinates are [y,x,degree].

    Y = np.zeros((n,m,l))
    k = 0

    for i in range(p+1):
        for j in range(p+1-i):
            Y[:,:,k] = (y**i)*(x**j)
            k += 1

    # Converting the 3-dimensional matrix to a 2-dimensional one, starting with y_0 for the first 20 rows spanning
    # x_0-x_19, then y_1 for another 20 rows, etc.

    # Design Matrix
    X = np.zeros((n * m, l))
    # Converting the function values to a vector
    Z = np.zeros(n * m)

    for i in range(n):
        for j in range(m):
            X[n*i+j,:] = Y[i,j,:]
            Z[n*i+j] = z[i,j]

    return X, Z

def MSE(y_data,y_model):
    n = np.size(y_model)
    return np.sum((y_data-y_model)**2)/n

def R2(y_data, y_model):
    n = np.size(y_model)
    return 1 - np.sum((y_data-y_model)**2)/np.sum((y_data-np.sum(y_data/n))**2)

ztilde = FrankeLinReg(5)

#print(f"MSE = {MSE(z,ztilde)}")
#print(f"R² = {R2(z,ztilde)}")

print(z[0,0:5])
print(ztilde[0,0:5])
