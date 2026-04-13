import numpy as np


class KNN(object):
    """
    kNN classifier object.
    """

    def __init__(self, k=9, task_kind="classification"):
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
    def predict(self, test_data):
            """
            Runs prediction on the test data.

            Arguments:
                test_data (np.array): test data of shape (M,D)
            Returns:
                test_labels (np.array): labels of shape (M,)
            """

            return np.apply_along_axis(func1d=kNN_one_example, axis=1, arr=test_data, training_features=self.train_data, training_labels=self.train_labels, k=self.k, task_kind=self.task_kind)
        

# --- Helper Functions ---

def kNN_one_example(unlabeled_example, training_features, training_labels, k, task_kind):
    """Returns the label of a single unlabelled example."""

    # Compute distances
    distances = euclidean_dist(unlabeled_example, training_features)

    # Find neighbors
    nn_indices = find_k_nearest_neighbors(k, distances)

    # Get neighbors' labels
    neighbor_labels = training_labels[nn_indices]

    # Pick the best label based on the task kind
    best_label = predict_label(neighbor_labels, task_kind)

    return best_label

    
def euclidean_dist(example, training_examples):
    """Compute the Euclidean distance between a single example
    vector and all training_examples."""
    return np.sqrt(((training_examples - example) ** 2).sum(axis=1))   


def find_k_nearest_neighbors(k, distances):
    """Find the indices of the k smallest distances from a list of distances."""
    indices = np.argsort(distances)[:k]
    return indices
    

def predict_label(neighbor_labels, task_kind):
    """Return the prediction based on whether it is classification or regression."""
    
    if task_kind == "classification":
        return np.argmax(np.bincount(neighbor_labels.astype(int)))
        
    elif task_kind == "regression":
        return np.mean(neighbor_labels)
        
    else:
        raise ValueError(f"Unknown task_kind: {task_kind}")