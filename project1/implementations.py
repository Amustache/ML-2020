import numpy as np
from numpy.linalg import inv
from tqdm import tqdm

def mini_batch(y, tx, size, num):
    """ Generate random batches from the given data.
    
    Args:
        y: Class labels.
        tx: Features as a NxM matrix.
        size: Size of batches.
        num: Number of batches.
    Yields:
        labels and data for a batch.

    Usage:
        To iterate through all batches use the following for loop :

        for b_y, b_tx in mini_batch(y, tx, size, num):
    """
    N = len(y)
    # Get random indices
    batch_ind = np.random.permutation(np.arange(N))
    for batch_id in range(num):
        s_id = int(batch_id*size)
        e_id = int(s_id+size)
        ids = batch_ind[s_id:e_id]
        yield y[ids], tx[ids]

def k_fold_batch(y, tx, k):
    """ Generate batches for k-fold.

    Args:
        y: Class labels.
        tx: Features as a NxM matrix.
        k: Number of folds.
    Return:
        y_train: k-1 batches of labels for training.
        tx_train: k-1 batches of data for training.
        y_test: 1 batch of labels for testing.
        tx: 1 batch of data for testing
    """
    N = len(y)
    if N%k != 0:
        print("K-fold error: Size of tX must be multiple of k.")
        return
    batch_size = int(N/k)
    # Get random indices
    batch_ind = np.arange(N)
    y_train = []
    tx_train = []
    for batch_id in range(k-1):
        s_id = int(batch_id*batch_size)
        e_id = int(s_id+batch_size)
        y_train.append(y[s_id:e_id].copy())
        tx_train.append(tx[s_id:e_id].copy())
    y_test = y[N-batch_size:N]
    tx_test = tx[N-batch_size:N]
    return y_train, tx_train, y_test, tx_test

def mae_loss(y, tx, w):
    """ Compute the MAE loss. """
    e = y - tx.dot(w)
    return np.sum(np.abs(e))/(2*len(y))


def mse_loss(y, tx, w):
    """ Compute the MSE loss. """
    e = y - tx.dot(w)
    return np.sum(np.square(e))/(2*len(y))

def cross_validate_ridge_regression(y, tx, k, l_st, l_en, l_space):
    """ K-fold cross-validation for selecting lambda for ridge regression.
    
    Args:
        y: class labels.
        tx: Features as a NxM matrix.
        k: Number of folds.
        l_st: Start of logspace for lambdas to test.
        l_en: End of logspace for lambdas to test.
        l_space: Number of lambdas to test in the logspace.

    Returns:
        The lambda with the lowest mean error for cross-validation with k folds.
    """
    size_batches = tx/k
    y_train_set, tx_train_set, y_test, tx_test = k_fold_batch(y, tx, k)
    min_loss = 999999999999
    best_l = 0
    tested_loss = []
    tested_lambda = []
    for l in tqdm(np.logspace(l_st, l_en, num=l_space)):
        tmp_loss = []
        for i in range(len(y_train_set)):
            b_y = y_train_set[i]
            b_tx = tx_train_set[i]
            w, _ = ridge_regression(b_y, b_tx,l)
            tmp_loss.append(mse_loss(y_test, tx_test, w))
        mean_loss = np.mean(tmp_loss)
        # Loss and lambda used for plot.
        tested_loss.append(mean_loss)
        tested_lambda.append(l)
        if mean_loss < min_loss:
            min_loss = mean_loss
            best_l = l
    plt.semilogx(tested_lambda, tested_loss, "*r")
    return best_l

def cross_validate_reg_logistic_regression(y, tx, initial_w, max_iters, gamma, k, l_st, l_en, l_space):
    """ K-fold cross-validation for selecting lambda for regularized logistic regression.
    
    Args:
        y: class labels.
        tx: Features as a NxM matrix.
        initial_w: Initial weights vector.
        max_iters: Number of steps.
        gamma: Step size.
        k: Number of folds.
        l_st: Start of logspace for lambdas to test.
        l_en: End of logspace for lambdas to test.
        l_space: Number of lambdas to test in the logspace.

    Returns:
        The lambda with the lowest mean error for cross-validation with k folds.
    """
    size_batches = tx/k
    y_train_set, tx_train_set, y_test, tx_test = k_fold_batch(y, tx, k)
    min_loss = 999999999999
    best_l = 0
    tested_loss = []
    tested_lambda = []
    for l in tqdm(np.logspace(l_st, l_en, num=l_space)):
        tmp_loss = []
        for i in range(len(y_train_set)):
            b_y = y_train_set[i]
            b_tx = tx_train_set[i]
            w, _ = reg_logistic_regression(b_y, b_tx, l, initial_w, max_iters, gamma)
            tmp_loss.append(mse_loss(y_test, tx_test, w))
        mean_loss = np.mean(tmp_loss)
        # Loss and lambda used for plot.
        tested_loss.append(mean_loss)
        tested_lambda.append(l)
        if mean_loss < min_loss:
            min_loss = mean_loss
            best_l = l
    plt.semilogx(tested_lambda, tested_loss, "*r")
    return best_l

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
    M, N = tx.shape
    w = inv(np.add(tx.transpose().dot(tx), 2*M*lambda_ * np.identity(N))).dot(tx.transpose()).dot(y)
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
    for i in range(max_iters):
        loss = np.sum(np.log(1 + np.exp(tx.dot(w))) - y * (tx.dot(w))) + (lambda_/2) * np.sum(np.power(np.linalg.norm(w), 2))
        grad = tx.transpose().dot(logistic_function(tx.dot(x))-y) + lambda_ * np.sum(np.linalg.norm(w))
        w = w - gamma*grad
    return w, loss
