import numpy as np


class KNN(object):
    """
    kNN classifier object.
    """

    def __init__(self, k=1, task_kind="classification"):
        """
        Call set_arguments function of this class.
        """
        self.k = k
        self.task_kind = task_kind

    def fit(self, training_data, training_labels):
        """
        Trains the model, returns predicted labels for training data.

        Hint: Since KNN does not really have parameters to train, you can try saving
        the training_data and training_labels as part of the class. This way, when you
        call the "predict" function with the test_data, you will have already stored
        the training_data and training_labels in the object.

        Arguments:
            training_data (np.array): training data of shape (N,D)
            training_labels (np.array): labels of shape (N,)
        Returns:
            pred_labels (np.array): labels of shape (N,)
        """
        self.train_data = training_data
        self.train_labels = training_labels

        pred_labels = self.predict(training_data)
        
        return pred_labels
    
    def euclidean_dist(self, example, training_examples):
        """Compute the Euclidean distance between a single example
        vector and all training_examples.

        Inputs:
            example: shape (D,)
            training_examples: shape (NxD) 
        Outputs:
            euclidean distances: shape (N,)
        """
        return np.sqrt(((training_examples - example) ** 2).sum(axis=1))
    

    def find_k_nearest_neighbors(self, k, distances):
        """ Find the indices of the k smallest distances from a list of distances.
            Tip: use np.argsort()

        Inputs:
            k: integer
            distances: shape (N,) 
        Outputs:
            indices of the k nearest neighbors: shape (k,)
        """
        indices = np.argsort(distances)[:k]
        return indices
    

    def predict_label(self, neighbor_labels):
        """Return the most frequent label in the neighbors'.
    
        Inputs:
            neighbor_labels: shape (N,) 
        Outputs:
            most frequent label
        """
        if self.task_kind == "regression":
            return np.mean(neighbor_labels)
        return np.argmax(np.bincount(neighbor_labels.astype(int)))

    def predict(self, test_data):
        """
        Runs prediction on the test data.

        Arguments:
            test_data (np.array): test data of shape (N,D)
        Returns:
            test_labels (np.array): labels of shape (N,)
        """
        test_labels = np.empty(test_data.shape[0], dtype=self.train_labels.dtype)

        for i, example in enumerate(test_data):
            distances = self.euclidean_dist(example, self.train_data)
            nn_indices = self.find_k_nearest_neighbors(self.k, distances)
            neighbor_labels = self.train_labels[nn_indices]
            test_labels[i] = self.predict_label(neighbor_labels)

        return test_labels
