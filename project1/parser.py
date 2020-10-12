import csv
import numpy as np

def read_csv(data_path):
    """ Return ids, data and predictions from csv file given a data_path. """
    data = np.genfromtxt(data_path, delimiter=",", skip_header=1)
    # Predicitons are string so must extract them separately to avoid NaN results
    y_str = np.genfromtxt(data_path, delimiter=",", skip_header=1, dtype=str, usecols=1)
    
    # Extract ids, x and y
    ids = data[:, 0].astype(np.int)
    x = data[:, 2:]

    # Convert string prediction to binary 0 or 1
    y = np.ones(len(y_str))
    y[np.where(y_str=='b')] = 0

    return y, x, ids 
