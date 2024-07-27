import numpy as np
import util
import sys
from random import random
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix
import pandas as pd

### NOTE : You need to complete logreg implementation first! If so, make sure to set the regularization weight to 0.

# Character to replace with sub-problem letter in plot_path/save_path
WILDCARD = 'X'
# Ratio of class 0 to class 1
kappa = 0.1


def plot_decision_boundary(X, y, model, title):
    # Plot data points
    plt.figure(figsize=(10, 6))
    plt.scatter(X[y == 0][:, 0], X[y == 0][:, 1], label='Class 0', alpha=0.6)
    plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], label='Class 1', alpha=0.6)

    # Plot decision boundary
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, alpha=0.2, cmap=plt.cm.Paired)

    plt.title(title)
    plt.xlabel('x_1')
    plt.ylabel('x_2')
    plt.legend()
    plt.show()

def load_data(file_path):
    data = pd.read_csv(file_path)
    X = data[['x_1', 'x_2']].values
    y = data['y'].values
    return X, y

def reweight_data(X, y, kappa):
    X_reweighted = np.copy(X)
    y_reweighted = np.copy(y)

    # Repeat each positive example 1/kappa times
    positive_indices = np.where(y == 1)[0]
    for index in positive_indices:
        for _ in range(int(1 / kappa) - 1):
            X_reweighted = np.vstack((X_reweighted, X[index]))
            y_reweighted = np.hstack((y_reweighted, y[index]))

    return X_reweighted, y_reweighted

def evaluate_model(X_val, y_val, model, save_path):
    # Predict classes on the validation set
    y_val_pred = model.predict(X_val)

    # Compute metrics
    accuracy = accuracy_score(y_val, y_val_pred)
    balanced_accuracy = balanced_accuracy_score(y_val, y_val_pred)
    cm = confusion_matrix(y_val, y_val_pred)
    TN, FP, FN, TP = cm.ravel()
    A0 = TN / (TN + FP)
    A1 = TP / (TP + FN)

    # Save predicted probabilities
    y_val_pred_proba = model.predict_proba(X_val)[:, 1]
    np.savetxt(save_path, y_val_pred_proba)
    return accuracy, balanced_accuracy, A0, A1


def main(train_path, validation_path, save_path):
    """Problem 2: Logistic regression for imbalanced labels.

    Run under the following conditions:
        1. naive logistic regression
        2. upsampling minority class

    Args:
        train_path: Path to CSV file containing training set.
        validation_path: Path to CSV file containing validation set.
        save_path: Path to save predictions.
    """
    output_path_naive = save_path.replace(WILDCARD, 'naive')
    output_path_upsampling = save_path.replace(WILDCARD, 'upsampling')

    # *** START CODE HERE ***

    #X_train, y_train = util.load_dataset(train_path)
    #X_val, y_val = util.load_dataset(validation_path)

    X_train, y_train = load_data(train_path)
    X_val, y_val = load_data(validation_path)
    X_train, y_train = load_data(train_path)
    X_val, y_val = load_data(validation_path)

    # Part (b): Vanilla logistic regression
    model_vanilla = LogisticRegression()
    model_vanilla.fit(X_train, y_train)

    # Evaluate vanilla logistic regression
    accuracy, balanced_accuracy, A0, A1 = evaluate_model(X_val, y_val, model_vanilla, output_path_naive)
    print('Vanilla Logistic Regression:')
    print(f'Accuracy: {accuracy}')
    print(f'Balanced Accuracy: {balanced_accuracy}')
    print(f'Accuracy for class 0 (A0): {A0}')
    print(f'Accuracy for class 1 (A1): {A1}')

    # Plot decision boundary for vanilla logistic regression
    plot_decision_boundary(X_val, y_val, model_vanilla, 'Vanilla Logistic Regression')

    # Part (d): Re-weighting minority class
    X_train_reweighted, y_train_reweighted = reweight_data(X_train, y_train, kappa)

    # Train logistic regression on re-weighted data
    model_reweighted = LogisticRegression()
    model_reweighted.fit(X_train_reweighted, y_train_reweighted)

    # Evaluate re-weighted logistic regression
    accuracy_rw, balanced_accuracy_rw, A0_rw, A1_rw = evaluate_model(X_val, y_val, model_reweighted,
                                                                     output_path_upsampling)
    print('Re-weighted Logistic Regression:')
    print(f'Accuracy: {accuracy_rw}')
    print(f'Balanced Accuracy: {balanced_accuracy_rw}')
    print(f'Accuracy for class 0 (A0): {A0_rw}')
    print(f'Accuracy for class 1 (A1): {A1_rw}')

    # Plot decision boundary for re-weighted logistic regression
    plot_decision_boundary(X_val, y_val, model_reweighted, 'Re-weighted Logistic Regression')
    # Repeat minority examples 1 / kappa times

    # *** END CODE HERE


if __name__ == '__main__':
    main(train_path='train.csv',
         validation_path='validation.csv',save_path='imbalanced_X_pred.txt')
