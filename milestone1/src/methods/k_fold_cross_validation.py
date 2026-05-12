import numpy as np
from src.methods.knn import KNN
from src.methods.logistic_regression import LogisticRegression
from src.utils import accuracy_fn, mse_fn

def KFold_cross_validation_KNN(X, Y, K, k, task):
    N = X.shape[0]
    indices = np.arange(N)
    fold_size = N // K

    scores = []
    
    for fold_index in range(K) :
        start = fold_index * fold_size
        end = (fold_index + 1) * fold_size if fold_index != K - 1 else N

        validation_indices = indices[start:end]
        train_indices = np.setdiff1d(indices, validation_indices, assume_unique = True)

        X_train_fold = X[train_indices, :]
        Y_train_fold = Y[train_indices]
        X_validation_fold = X[validation_indices, :]
        Y_validation_fold = Y[validation_indices]

        model = KNN(k = k, task_kind = task)
        model.fit(X_train_fold, Y_train_fold)

        Y_pred = model.predict(X_validation_fold)

        if task == "classification" :
            score = accuracy_fn(Y_pred, Y_validation_fold)
        else :
            score = mse_fn(Y_pred, Y_validation_fold)

        scores.append(score)
    return np.mean(scores)



 

def run_cv_for_hyperparam_KNN(X, Y, K, k_list, task):
    model_performance = [] 
    for k in k_list:
        model_performance.append(KFold_cross_validation_KNN(X, Y, K, k, task))      
    return model_performance




def KFold_cross_validation_Log_Reg(X, Y, K, lr, max_iters):
    N = X.shape[0]
    indices = np.arange(N)
    fold_size = N // K

    scores = []
    
    for fold_index in range(K) :
        start = fold_index * fold_size
        end = (fold_index + 1) * fold_size if fold_index != K - 1 else N

        validation_indices = indices[start:end]
        train_indices = np.setdiff1d(indices, validation_indices, assume_unique = True)

        X_train_fold = X[train_indices, :]
        Y_train_fold = Y[train_indices]
        X_validation_fold = X[validation_indices, :]
        Y_validation_fold = Y[validation_indices]

        model = LogisticRegression(lr = lr, max_iters = max_iters)
        model.fit(X_train_fold, Y_train_fold)

        Y_pred = model.predict(X_validation_fold)

        score = accuracy_fn(Y_pred, Y_validation_fold)
        
        scores.append(score)
    return np.mean(scores)

def run_cv_for_hyperparam_Log_Reg(X, Y, K, lr_list, max_iters):
    model_performance = [] 
    for lr in lr_list:
        model_performance.append(KFold_cross_validation_Log_Reg(X, Y, K, lr, max_iters))      
    return model_performance
