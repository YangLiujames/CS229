import numpy as np
import util
import matplotlib.pyplot as plt


def main(train_path, save_path):
    """Problem: Logistic regression with gradient descent.

    Args:
        train_path: Path to CSV file containing dataset for training.
        save_path: Path to save outputs; visualizations, predictions, etc.
    """
    x_train, y_train = util.load_csv(train_path, add_intercept=True)

    # *** START CODE HERE ***
    # Train a logistic regression classifier
    # Plot decision boundary on top of training set.
    # Use save_path argument to save various visualizations for your own reference.

    # Train a logistic regression classifier
    clf = LogisticRegression()
    clf.fit(x_train, y_train)

    # Plot decision boundary on top of training set
    plot_decision_boundary(x_train, y_train, clf.theta, save_path)
    plot_loss(clf.loss_history)
    # *** END CODE HERE ***


class LogisticRegression:
    """Logistic regression using gradient descent.

    Example usage:
        > clf = LogisticRegression()
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """
    def __init__(self, learning_rate=0.0001, max_iter=100000, eps=1e-5,
                 theta_0=None, verbose=True):
        """
        Args:
            learning_rate: Step size for iterative solvers only.
            max_iter: Maximum number of iterations for the solver.
            eps: Threshold for determining convergence.
            theta_0: Initial guess for theta. If None, use the zero vector.
            verbose: Print loss values during training.
        """
        self.theta = theta_0
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.eps = eps
        self.verbose = verbose
        self.lambda_ = 0.01
        self.loss_history = []
        # *** START CODE HERE ***
        # *** END CODE HERE ***

    def fit(self, x, y):
        """Run gradient descent to minimize J(theta) for logistic regression.

        Args:
            x: Training example inputs. Shape (n_examples, dim).
            y: Training example labels. Shape (n_examples,).
        """
        # *** START CODE HERE ***
        m, n = x.shape
        if self.theta is None:
            self.theta = np.zeros(n)

        for i in range(self.max_iter):
            z = np.dot(x, self.theta)
            h = 1 / (1 + np.exp(-z))
            gradient = np.dot(x.T, (h - y)) / m + self.lambda_ * self.theta / m
            theta_prev = self.theta
            self.theta -= self.learning_rate * gradient

            loss = self.loss(x, y)
            self.loss_history.append(loss)

            if self.verbose and i % 1000 == 0:
                print(f'Iteration {i}: Loss = {loss}, Theta = {self.theta}')

            if np.linalg.norm(self.theta - theta_prev, ord=1) < self.eps:
                break
        # *** END CODE HERE ***

    def predict(self, x):
        """Return predicted probabilities given new inputs x.

        Args:
            x: Inputs of shape (n_examples, dim).

        Returns:
            Outputs of shape (n_examples,).
        """
        # *** START CODE HERE ***
        z = np.dot(x, self.theta)
        return 1 / (1 + np.exp(-z))
        # *** END CODE HERE ***

    def loss(self, x, y):
        """Compute the logistic loss function."""
        z = np.dot(x, self.theta)
        h = 1 / (1 + np.exp(-z))
        return -np.mean(y * np.log(h) + (1 - y) * np.log(1 - h)) + self.lambda_ / 2 * np.dot(self.theta, self.theta)


def plot_decision_boundary(x, y, theta, save_path):
    """Plot the decision boundary."""
    plt.figure()

    # Plotting the points
    plt.scatter(x[:, 1], x[:, 2], c=y, cmap=plt.cm.Spectral)

    # Plotting the decision boundary
    x_values = [np.min(x[:, 1] - 2), np.max(x[:, 2] + 2)]
    y_values = - (theta[0] + np.dot(theta[1], x_values)) / theta[2]
    plt.plot(x_values, y_values, label='Decision Boundary')

    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.legend()
    plt.show()

def plot_loss(loss_history):
    plt.figure()
    plt.plot(loss_history)
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Loss over Iterations')
    plt.show()


if __name__ == '__main__':
    print('==== Training model on data set A ====')
    main(train_path='ds1_a.csv',
         save_path='logreg_pred_a.txt')

    print('\n==== Training model on data set B ====')
    main(train_path='ds1_b.csv',
         save_path='logreg_pred_b.txt')
