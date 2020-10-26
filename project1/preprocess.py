import numpy as np

def modifyCSV(tx, invalid=True, standardize=True):
    """ Modify the data to remove invalid values, -999, and/or standardize the data.

    Args:
        tx: Data to be modified
        invalid: Replace invalid values with mean of valid values
        standardize: Standardize values
    Return:
        Return modified data
    """
    res = tx.copy()
    if (invalid):
        res = invalidToMean(res)
    if (standardize):
        res = standardizeValues(res)
    return res
    

def standardizeValues(tx):
    """ Standardize values. """
    res = tx.copy()
    M,N = tx.shape
    means = np.mean(res, axis=0)
    std_dev = np.std(res, axis=0)
    std_tx = np.divide(np.subtract(res, means),std_dev)
    return std_tx


def standardize(x):
    """ Standardize the original data set.

    Args:
        x: Data set.
    Return:
        Standardized data set.
    """
    mean_x = np.mean(x, axis=0)
    x = x - mean_x
    std_x = np.std(x, axis=0)
    x = x / std_x
    return x, mean_x, std_x


def invalidToMean(tx):
    """ Change all -999 values with the mean of valid values of same column. """
    _,N = tx.shape
    res = tx.copy()
    tmp_tx = tx.copy()

    # Set all -999 values to NaN
    tmp_tx[tx == -999] = np.nan

    # Compute mean ignoring NaN values
    means = np.nanmean(tmp_tx, axis=0)
    
    for i in range(N):
        # For each column replace values -999 with the mean of valid values
        res[:,i][res[:,i] == -999] = means[i]

    return res

def displayFeatures(tx):
    M, N = tx.shape
    ptX = modifyCSV(tx)
    for i in range(2, N):
        ax = plt.subplot()
        plt.scatter(np.arange(M), ptX[:,i])
        plt.title("Feature "+str(i-1))
        plt.show()

def computeCorrelation(tx):
    M, N = tx.shape
    ptX = modifyCSV(tx)
    for i in range(2, N-1):
        for j in range(i+1, N):
            mean_x = np.mean(ptX[:,i])
            mean_y = np.mean(ptX[:,j])
            r = np.divide(np.sum(np.multiply(np.subtract(ptX[:,i], mean_x), np.subtract(ptX[:,j], mean_y))), np.multiply(np.sqrt(np.sum(np.power(np.subtract(ptX[:,i], mean_x),2))), np.sqrt(np.sum(np.power(np.subtract(ptX[:,j], mean_y),2)))))
            if (r > 0.9):
                print("Correlation between "+str(i-1)+" and "+str(j-1)+" :")
                print(r)
            if (r < -0.9):
                print("Correlation between "+str(i-1)+" and "+str(j-1)+" :")
                print(r)
