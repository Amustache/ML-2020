from sys import argv
import os
import pickle

from proj1_helpers import *
from implementations import *
from preprocess import *
from metrics import *


def model(y, tX, k=10, l_st=-3.5, l_en=-2.5, l_space=100, lambda_=None, save=None):
    """ Model is using ridge regression with different parameters and auto tunning"""
    # Data preprocessing
    print("Data preprocessing.")
    print("-" * 10)

    # Standardization
    # tX_stand, tx_mean, tx_std = standardize(tX)
    tX_stand = standardizeValues(tX)

    # Replacing odd (outliers) values with mean, column-wize
    tX_stand = invalidToMean(tX_stand)
    # for i in range(tX_stand.shape[1]):
    #     tX_stand[:, i][np.where(tX_stand[:, i] == -999)] = tx_mean[i]

    # Tunning
    if not lambda_:
        lambda_ = cross_validate_ridge_regression(y, tX_stand, k, l_st, l_en, l_space)

    # Learning
    weights, loss = ridge_regression(y, tX, lambda_)

    # Save weights
    if save:
        weight_path = os.path.join(save, "weights_{}.p".format(str(datetime.now())), "wb")
        print("Saving weights to {}.".format(weight_path))
        print("-" * 10)
        pickle.dump(weights, open(weight_path))

    return weights, loss


if __name__ == "__main__":
    if len(argv) != 4:
        raise ValueError("Error: Not enough arguments. Use: `python run.py <input_train_data>.csv <input_test_data>.csv <output_data>.csv`")

    if argv[1][-4:] != '.csv' or argv[2][-4:] != '.csv' or argv[3][-4:] != '.csv':
        raise ValueError("Error: Wrong filename. Use: `python run.py <input_train_data>.csv <input_test_data>.csv <output_data>.csv`")

    name, train_path, test_path, output_path = argv

    # Import data
    print("Importing data from {}.".format(train_path))
    print("-" * 10)
    y, tX, ids = load_csv_data(train_path)

    # Generate model
    print("Generating model.")
    print("-" * 10)
    weights, loss = model(y, tX)

    # Generating predictions
    print ("Generating predictions for {}.".format(test_path))
    print("-" * 10)
    y_test, tX_test, ids_test = load_csv_data(test_path)
    y_pred = predict_labels(weights, tX_test)

    # Metrics
    F1, accuracy = metrics(y_test, y_pred)
    print("F1-score: {}, accuracy: {}.".format(F1, accuracy))
    print("-" * 10)

    # Export predictions
    print("Exporting predictions to {}.".format(output_path))
    print("-" * 10)
    create_csv_submission(ids_test, y_pred, output_path)
