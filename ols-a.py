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

def FrankeLinReg(x,y,z,p):

    X = Mesh2Des(x,y,p)
    Z = Mesh2Vec(z)

    #print(X.shape)

    beta = np.linalg.inv(X.T @ X) @ (X.T) @ Z

    #Ztilde = X @ beta

    #print(beta)
    #print(Ztilde.shape)

    # Converting the prediction ztilde back to a meshgrid

    #n, m = x.shape[0],x.shape[1]

    #ztilde = Vec2Mesh(Ztilde,n, m)

    #print(ztilde)

    return beta

def Vec2Mesh(Z, n, m):
    """
    Takes a vector Z and converts it into an n x m matrix, row by row.
    """
    z = np.zeros((n,m))
    if n*m != len(Z):
        print("oh fuck")

    for i in range(n):
        z[i,:] = Z[m * i : m * i + m]
    return z

def Mesh2Vec(z):
    """
    Converts a meshgrid into a vector.
    """
    n, m = z.shape[0], z.shape[1]
    # Converting the function values to a vector
    Z = np.zeros(n * m)

    for i in range(n):
        for j in range(m):
            Z[m*i+j] = z[i,j]

    return Z

def Mesh2Des(x,y,p):
    """
    Converts a meshgrid x, y into a design matrix X of polynomials
    in x, y up to degree p.
    """
    n, m = x.shape[0], x.shape[1]

    # Number of predictors
    l = int((p+1) * (p+2) / 2)

    # First making a 3-dimensional design matrix utilizing the meshgrid. Coordinates are [y,x,degree].
    # The design matrix is an n x m x l matrix, where l is the number of predictors = (p+1)*(p+2)/2
    # for polynomials of degree p, with two variables.

    Y = np.zeros((n,m,l))
    k = 0

    for i in range(p+1):
        for j in range(p+1-i):
            Y[:,:,k] = (y**i)*(x**j)
            k += 1

    # Converting the 3-dimensional matrix to a 2-dimensional one, starting with y_0 for the first m rows spanning
    # x_0-x_m-1, then y_1 for another m rows, etc.

    # Design Matrix
    X = np.zeros((n * m, l))

    for i in range(n):
        for j in range(m):
            X[n*i+j,:] = Y[i,j,:]

    return X

def MSE(y_data,y_model):
    n = np.size(y_model)
    return np.sum((y_data-y_model)**2)/n

def R2(y_data, y_model):
    n = np.size(y_model)
    return 1 - np.sum((y_data-y_model)**2)/np.sum((y_data-np.sum(y_data/n))**2)

def makePred(x,y,p,beta):
    """
    Makes a prediction with a given predictor vector beta on the mesh
    grid x, y.
    """
    (n, m) = x.shape
    X = Mesh2Des(x,y,p)
    #print(n)
    #print(m)
    Z_pred = X @ beta
    z_pred = Vec2Mesh(Z_pred, n, m)
    print(f"Predshape = {z_pred.shape}")
    return z_pred

x_train, x_test, y_train, y_test, z_train, z_test = train_test_split(x, y, z, test_size=0.01)

p = 5
beta = FrankeLinReg(x_train, y_train, z_train, p)
z_pred = makePred(x_test, y_test, p,beta)

np.set_printoptions(precision=3, suppress = True)
print(f"MSE = {MSE(z_test,z_pred):.3f}")
print(f"R² = {R2(z_test,z_pred):.3f}")
print(beta)
print(z_test[0,:])
print(z_pred[0,:])
#print(Mesh2Des(x_train,y_train,p)[1])
