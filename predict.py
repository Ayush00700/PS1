import numpy as np
from L_model_forward import L_model_forward
import csv


def predict(X, y, parameters):
    """
    This function is used to predict the results of a  L-layer neural network.

    Arguments:
    X -- data set of examples you would like to label
    parameters -- parameters of the trained model

    Returns:
    p -- predictions for the given dataset X
    """

    m = X.shape[1]
    n = len(parameters) // 2  # number of layers in the neural network
    p = np.zeros((1, m))

    # Forward propagation
    probas, caches = L_model_forward(X, parameters)

    # convert probas to 0/1 predictions
    for i in range(0, probas.shape[1]):
        if probas[0, i] > 0.5:
            p[0, i] = 1
        else:
            p[0, i] = 0

    # print results
    print("predictions: " + str(p))
    # p.forEach(e= > console.log(e))
    # y.forEach(e= > console.log(e))

    print("true labels: " + str(y))
    print("Accuracy: " + str(np.sum((p == y)/m)))
    filename = "Predictions.csv"
    with open(filename, 'w') as csvfile:
        # creating a csv writer object
        csvwriter = csv.writer(csvfile)

        # writing the fields
        csvwriter.writerow(p)

        # writing the data rows
        csvwriter.writerows(y)
    return p
