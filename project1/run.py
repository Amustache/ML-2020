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

    # Replace invalid values with mean of valid values and standardize
    tX = modifyCSV(tX)

    # Remove features due to correlation
    tX = featureRemoval(tX)

    # Augment features
    M,N = tX.shape
    # tX = np.concatenate((np.ones((M,1)),tX, np.sin(tX), np.cos(tX)), axis=1)

    # Tunning
    if not lambda_:
        lambda_ = cross_validate_ridge_regression(y, tX, k, l_st, l_en, l_space)

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

    # Metrics
    F1, accuracy = metrics(y, predict_labels(weights, tX))
    print("F1-score: {}, accuracy: {}.".format(F1, accuracy))
    print("-" * 10)

    # Generating predictions
    print ("Generating predictions for {}.".format(test_path))
    print("-" * 10)
    _, tX_test, ids_test = load_csv_data(test_path)
    y_pred = predict_labels(weights, tX_test)

    # Export predictions
    print("Exporting predictions to {}.".format(output_path))
    print("-" * 10)
    create_csv_submission(ids_test, y_pred, output_path)
