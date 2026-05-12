import numpy as np

from ..utils import get_n_classes, label_to_onehot, onehot_to_label


class LogisticRegression(object):
    """
    Logistic regression classifier.
    """

    def __init__(self, lr, max_iters=500):
        """
        Initialize the new object (see dummy_methods.py)
        and set its arguments.

        Arguments:
            lr (float): learning rate of the gradient descent
            max_iters (int): maximum number of iterations
        """
        self.lr = lr
        self.max_iters = max_iters
        self.weights = None
        self.nb_classes = None

    def softmax(self, raw_outputs) : 
        exp_raw_outputs_shifted = np.exp(raw_outputs - np.max(raw_outputs, axis = 1, keepdims = True))
        sum_exp = np.sum(exp_raw_outputs_shifted, axis = 1, keepdims = True)
        return exp_raw_outputs_shifted / sum_exp

    def fit(self, training_data, training_labels):
        """
        Trains the model, returns predicted labels for training data.

        Arguments:
            training_data (np.array): training data of shape (N,D)
            training_labels (np.array): regression target of shape (N,)
        Returns:
            pred_labels (np.array): target of shape (N,)
        """
        ##
        ###
        #### WRITE YOUR CODE HERE!
        ###
        ##

        nb_samples, nb_dimensions = training_data.shape
        self.nb_classes = get_n_classes(training_labels)
        self.weights = np.zeros((nb_dimensions, self.nb_classes))

        one_hot_labels = label_to_onehot(training_labels, self.nb_classes)

        for i in range(self.max_iters) :

            raw_output = training_data @ self.weights
            output_probas = self.softmax(raw_output)
            grad = training_data.T @ (output_probas - one_hot_labels) / nb_samples
            self.weights -= self.lr * grad

        pred_labels = onehot_to_label(output_probas)
        return pred_labels

    def predict(self, test_data):
        """
        Runs prediction on the test data.

        Arguments:
            test_data (np.array): test data of shape (N,D)
        Returns:
            pred_labels (np.array): labels of shape (N,)
        """
        ##
        ###
        #### WRITE YOUR CODE HERE!
        ###
        ##

        raw_output = test_data @ self.weights
        output_probas = self.softmax(raw_output)
        pred_labels = onehot_to_label(output_probas)   

        return pred_labels
