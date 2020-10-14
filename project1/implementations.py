import numpy as np
from numpy.linalg import inv
from tqdm import tqdm

def mini_batch(y, tx, size, num):
    N = len(y)
    # Get random indices
    batch_ind = np.random.permutation(np.arange(N))
    for batch_id in range(num):
        s_id = int(batch_id*size)
        e_id = int(s_id+size)
        ids = batch_ind[s_id:e_id]
        yield y[ids], tx[ids]

def mae_loss(y, tx, w):
    """ Compute the MAE loss. """
    e = y - tx.dot(w)
    return np.sum(np.abs(e))/(2*len(y))


def mse_loss(y, tx, w):
    """ Compute the MSE loss. """
    e = y - tx.dot(w)
    return np.sum(np.square(e))/(2*len(y))


""" Requested ML methods. """
def least_square_GD(y, tx, initial_w, max_iters, gamma):
    """ Linear regression using gradient descent. """
    w = initial_w
    for i in tqdm(range(max_iters)):
        # Compute error vector
        e = y-tx.dot(w)
        # Compute gradient
        grad = (-1/len(y)) * tx.transpose().dot(e)
        # Compute w(t+1)
        w = w - gamma*grad
        # Compute loss
        loss = mse_loss(y, tx, w)
    return w, loss

def least_square_SGD(y, tx, initial_w, max_iters, gamma):
    """ Linear regression using stochastic gradient descent. """
    w = initial_w
    for i in tqdm(range(max_iters)):
        size = len(y)/20
        grad = 0
        num = 5
        for b_y, b_tx in mini_batch(y, tx, size, num):
            e = b_y-b_tx.dot(w)
            grad += (-1/len(b_y)) * b_tx.transpose().dot(e)
        w = w - gamma*grad
        loss = mse_loss(y, tx, w)
    return w, loss

def least_squares(y, tx):
    """ Least squares using normal equations. """
    w = inv(tx.transpose().dot(tx)).dot(tx.transpose()).dot(y)
    loss = mse_loss(y, tx, w)
    return w, loss

def ridge_regression(y, tx, lambda_):
    """ Ridge regression using normal equations. """
    N = len(y)
    w = np.inv(np.add(tx.transpose().dot(tx), 2*N*lambda_ * np.identity(N))).dot(tx.transpose()).dot(y)
    loss = mse_loss(y, tx, w)
    return w, loss

def logistic_function(z):
    """ The logistic function sigma. """
    return np.exp(z)/(1+np.exp(z))

def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """ Logistic regression using gradient descent or SGD. """
    for i in range(max_iters):
        loss = np.sum(np.log(1 + np.exp(tx.dot(w))) - y * (tx.dot(w)))
        grad = tx.transpose().dot(logistic_function(tx.dot(x))-y)
    w = w - gamma*grad
    return w, loss

def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    """ Regularized logistic regression using gradient descent or SGD. """
    return w, loss
