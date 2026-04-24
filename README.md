# CS-233: Introduction to Machine Learning - Project Milestone 1

**Authors:** Paul Quignodon, Lennox Victor Ruffiner, Adam Madjid  
**Institution:** EPFL

## 📝 Project Overview
This project evaluates traditional machine learning algorithms on the **Gaming and Mental Health Dataset** (2000 samples, 13 features). We implemented three algorithms from scratch to address two main tasks:
* **Regression:** Predicting a continuous addiction score (0-10).
* **Classification:** Predicting discrete addiction levels (Low, Medium, High).

## 🚀 Implemented Algorithms
* **Linear Regression:** Closed-form pseudo-inverse solution.
* **Logistic Regression:** Multiclass softmax classifier trained via gradient descent.
* **K-Nearest Neighbors (KNN):** Distance-based classification and regression.

## 📊 Final Performance Results
Linear Regression and Logistic Regression significantly outperformed KNN across both tasks.

| Method | Metric | Validation | Test |
| :--- | :--- | :--- | :--- |
| **Linear Regression** | MSE ↓ | **0.931** | **0.994** |
| KNN Reg. ($k=9$) | MSE ↓ | 1.947 | 1.865 |
| **LogReg ($\eta=0.3, 500$)** | Acc. (%) ↑ | **86.04** | **88.25** |
| KNN Cls. ($k=9$) | Acc. (%) ↑ | 80.63 | 79.75 |
| **LogReg ($\eta=0.3, 500$)** | Macro-F1 ↑ | **0.727** | **0.819** |

## ⏱️ Runtime Analysis
Measured on 1600 training samples and 400 test samples.

| Method | Train Time | Predict (400) |
| :--- | :--- | :--- |
| Linear Regression | 0.43 ms | 0.01 ms |
| Logistic Reg. | 119 ms | 0.09 ms |
| KNN ($k=9$) | $\approx$ 0 ms | ~35 ms |
