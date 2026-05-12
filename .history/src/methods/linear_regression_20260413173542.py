import numpy as np

from ..utils import append_bias_term


class LinearRegression(object):
    """
    Linear regression.
    """

    def __init__(self):
        """
        Initialize the new object (see dummy_methods.py)
        and set its arguments.
        """
        self.weights = None  # shape (D+1,) including bias

    def fit(self, training_data, training_labels):
        """
        Trains the model, returns predicted labels for training data.

        Hint: You can use the closed-form solution for linear regression
        (with or without regularization). Remember to handle the bias term.

        Arguments:
            training_data (np.array): training data of shape (N,D)
            training_labels (np.array): regression target of shape (N,)
        Returns:
            pred_labels (np.array): target of shape (N,)
        """
        # Augment data with a bias column of ones: shape (N, D+1)
        X = append_bias_term(training_data)

        # Closed-form least-squares solution with pseudo-inverse for stability.
        self.weights = np.linalg.pinv(X) @ training_labels

        pred_labels = X @ self.weights
        return pred_labels

    def predict(self, test_data):
        """
        Runs prediction on the test data.

        Arguments:
            test_data (np.array): test data of shape (N,D)
        Returns:
            pred_labels (np.array): labels of shape (N,)
        """
        X = append_bias_term(test_data)
        pred_labels = X @ self.weights
        return pred_labels
