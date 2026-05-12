import argparse
import numpy as np

from src.methods.dummy_methods import DummyClassifier
from src.methods.logistic_regression import LogisticRegression
from src.methods.linear_regression import LinearRegression
from src.methods.knn import KNN
from src.utils import normalize_fn, append_bias_term, accuracy_fn, macrof1_fn, mse_fn
from src.methods.k_fold_cross_validation import run_cv_for_hyperparam_KNN, run_cv_for_hyperparam_Log_Reg
import os

np.random.seed(100)


def main(args):
    """
    The main function of the script.

    Arguments:
        args (Namespace): arguments that were parsed from the command line (see at the end
                          of this file). Their value can be accessed as "args.argument".
    """


    dataset_path = args.data_path
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found at {dataset_path}")

    ## 1. We first load the data.

    feature_data = np.load(dataset_path, allow_pickle=True)
    train_features, test_features, train_labels_reg, test_labels_reg, train_labels_classif, test_labels_classif = (
        feature_data['xtrain'],feature_data['xtest'],feature_data['ytrainreg'],
        feature_data['ytestreg'],feature_data['ytrainclassif'],feature_data['ytestclassif']
    )

    ## 2. Then we must prepare it. This is where you can create a validation set,
    #  normalize, add bias, etc.

    # Make a validation set (it can overwrite xtest, ytest)
    if not args.test:
        ### WRITE YOUR CODE HERE
        N = train_features.shape[0]
        indices = np.random.permutation(N)

        split_index = int(0.7 * N) # 70% train and 30% validation

        train_indices = indices[:split_index]
        validation_indices = indices[split_index :]

        validation_features = train_features[validation_indices]
        validation_labels_reg = train_labels_reg[validation_indices]
        validation_labels_classif = train_labels_classif[validation_indices]


        train_features = train_features[train_indices]
        train_labels_reg = train_labels_reg[train_indices]
        train_labels_classif = train_labels_classif[train_indices]
        
        test_features = validation_features
        test_labels_reg = validation_labels_reg
        test_labels_classif = validation_labels_classif
    

    # Compute train statistics per feature (column-wise).
    mean_train_features = np.mean(train_features, axis=0, keepdims=True)
    std_train_features = np.std(train_features, axis=0, keepdims=True)
    std_train_features[std_train_features == 0] = 1.0

    #normalizing every features of the data
    normalized_train_features = normalize_fn(train_features, mean_train_features, std_train_features)
    normalized_test_features = normalize_fn(test_features, mean_train_features, std_train_features)



    ## 3. Initialize the method you want to use.

    # Follow the "DummyClassifier" example for your methods
    if args.method == "dummy_classifier":
        method_obj = DummyClassifier(arg1=1, arg2=2)

    elif args.method == "knn":
        method_obj = KNN(k=args.K, task_kind=args.task)
        

    elif args.method == "logistic_regression":
        method_obj = LogisticRegression(lr=args.lr, max_iters=args.max_iters)

    elif args.method == "linear_regression":
        ### WRITE YOUR CODE HERE
        method_obj = LinearRegression()

    else:
        raise ValueError(f"Unknown method: {args.method}")
    

    ## BONUS : K-Fold Cross Validation for hyperparameter selection

    if args.use_cv :
        print("k-fold validation for the BONUS")

        X = normalized_train_features

        labels = train_labels_classif if args.task == "classification" else  train_labels_reg

        if args.method == "knn" :

            k_list = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21]
            cv_performances = run_cv_for_hyperparam_KNN(X = X, Y = labels, K = args.cv_nb_folds, k_list = k_list, task = args.task)

            print("CV results :", cv_performances)

            if args.task == "classification" :
                best_k = k_list[np.argmax(cv_performances)]
            else :
                best_k = k_list[np.argmin(cv_performances)]
                
                    
            print("best_k is :", best_k)

            method_obj = KNN(k = best_k, task_kind = args.task)

        elif args.method == "logistic_regression" : 

            X = append_bias_term(X)   

            lr_list = [0.001, 0.01, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0]
            cv_performances = run_cv_for_hyperparam_Log_Reg(X = X, Y = labels, K = args.cv_nb_folds, lr_list = lr_list, max_iters = args.max_iters)

            print("CV results :", cv_performances)

            best_lr = lr_list[np.argmax(cv_performances)]
            print("best_lr is :", best_lr)

            method_obj = LogisticRegression(lr = best_lr, max_iters = args.max_iters)

        else : 
            print("Cross-validation not supported for the method", args.method)    






    ## 4. Train and evaluate the method
    model_train_features = normalized_train_features
    model_test_features = normalized_test_features

    # Add bias in the pipeline for linear and logistic regression.
    if args.method in ("logistic_regression", "linear_regression"):
        model_train_features = append_bias_term(model_train_features)
        model_test_features = append_bias_term(model_test_features)

    if args.task == "classification":
        assert args.method != "linear_regression", f"You should use linear regression as a regression method"
        # Fit the method on training data
        preds_train = method_obj.fit(model_train_features, train_labels_classif)

        # Predict on unseen data
        preds = method_obj.predict(model_test_features)

        # Report results: performance on train and valid/test sets
        acc = accuracy_fn(preds_train, train_labels_classif)
        macrof1 = macrof1_fn(preds_train, train_labels_classif)
        print(f"\nTrain set: accuracy = {acc:.3f}% - F1-score = {macrof1:.6f}")

        acc = accuracy_fn(preds, test_labels_classif)
        macrof1 = macrof1_fn(preds, test_labels_classif)
        print(f"Test set:  accuracy = {acc:.3f}% - F1-score = {macrof1:.6f}")

    elif args.task == "regression":
        assert args.method != "logistic_regression", f"You should use logistic regression as a classification method"
        # Fit the method on training data
        preds_train = method_obj.fit(model_train_features, train_labels_reg)

        # Predict on unseen data
        preds = method_obj.predict(model_test_features)

        # Report results: MSE on train and valid/test sets
        train_mse = mse_fn(preds_train, train_labels_reg)
        print(f"\nTrain set: MSE = {train_mse:.6f}")

        test_mse = mse_fn(preds, test_labels_reg)
        print(f"Test set:  MSE = {test_mse:.6f}")

    else:
        raise ValueError(f"Unknown task: {args.task}")

    ### WRITE YOUR CODE HERE if you want to add other outputs, visualization, etc.


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task",
        default="classification",
        type=str,
        help="classification / regression",
    )
    parser.add_argument(
        "--method",
        default="dummy_classifier",
        type=str,
        help="dummy_classifier / knn / logistic_regression / linear_regression",
    )
    parser.add_argument(
        "--data_path",
        default="data/features.npz",
        type=str,
        help="path to your dataset CSV file",
    )
    parser.add_argument(
        "--K",
        type=int,
        default=1,
        help="number of neighboring datapoints used for knn",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-5,
        help="learning rate for methods with learning rate",
    )
    parser.add_argument(
        "--max_iters",
        type=int,
        default=100,
        help="max iters for methods which are iterative",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="train on whole training data and evaluate on the test data, "
             "otherwise use a validation set",
    )
    # Feel free to add more arguments here if you need!

    parser.add_argument(
        "--use_cv",
        action="store_true",
        help="allows us to choose whether or not to use cross-validation",
    )

    parser.add_argument(
        "--cv_nb_folds",
        type=int,
        default=5,
        help="number of folds used for k-fold cross validation",
    )

    args = parser.parse_args()
    main(args)
