import numpy as np


def basic_metrics(y, y_pred):
    TP = np.sum(np.logical_and(y_pred == 1, y == 1))
    TN = np.sum(np.logical_and(y_pred == -1, y == -1))
    FP = np.sum(np.logical_and(y_pred == 1, y == -1))
    FN = np.sum(np.logical_and(y_pred == -1, y == 1))
    return TP, TN, FP, FN


def precision(y, y_pred):
    TP, TN, FP, FN = basic_metrics(y, y_pred)
    return TP / (TP + FP)


def TPR(y, y_pred):
    TP, TN, FP, FN = basic_metrics(y, y_pred)
    return TP / (TP + FN)


def FPR(y, y_pred):
    TP, TN, FP, FN = basic_metrics(y, y_pred)
    return FP / (FP + TN)


def accuracy(y, y_pred):
    TP, TN, FP, FN = basic_metrics(y, y_pred)
    return (TP + TN) / (TP + TN + FP + FN)


def F1(y, y_pred):
    p = precision(y, y_pred)
    r = TPR(y, y_pred)
    return 2 * (p * r) / (p + r)


def metrics(y, y_pred):
    return F1(y, y_pred), accuracy(y, y_pred)
