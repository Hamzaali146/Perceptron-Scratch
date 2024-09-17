import numpy as np

def stepFunction(x):
    """Step function would be use while training neurons and it is basically a hard limitor function which generate a output one if netsum value from Perceptron class fit funtion is greater than threshold. Thresold value by default we are setting to 0.5"""
    threshold=0.5
    return np.where(x > threshold , 1, 0)

class Perceptron:
    """This is the class for implementing Perceptron model which is basically derived from supervised Learning learning rate is 0.01 iteration would be 1000 activation function used in this a step function"""
    def __init__(self, learning_rate=0.01, n_iters=1000):
        self.lr = learning_rate
        self.n_iters = n_iters
        self.activation_func = stepFunction
        self.weights = None
        self.bias = None


    def fit(self, X, y):
        """In this method you will basically train the training set and assess errors to update weights and bias to acheive target value weights will only be updated when error is not equal to zero means it is predicting wrong outputs update bias and weights"""
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        y_ = np.where(y > 0 , 1, 0)
        for _ in range(self.n_iters):
            for idx, x_i in enumerate(X):
                linear_output = np.dot(x_i, self.weights) + self.bias
                y_predicted = self.activation_func(linear_output)
                error=y_[idx] - y_predicted
                if error!=0:
                    update = self.lr * (error)
                    self.weights += update * x_i
                    self.bias += update


    def predict(self, X):
        """In which you are going to predict o\\p against the updated weights from fit() functions"""
        linear_output = np.dot(X, self.weights) + self.bias
        y_predicted = self.activation_func(linear_output)
        return y_predicted



if __name__ == "__main__":
    from sklearn.datasets import make_classification
    import matplotlib.pyplot as plt
    from sklearn.model_selection import train_test_split
    from sklearn import datasets

    def accuracy(y_true, y_pred):
        accuracy = np.sum(y_true == y_pred) / len(y_true)
        return accuracy

    X, y = make_classification(n_samples=1000, n_features=10, n_classes=2, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=123
    )

    p = Perceptron(learning_rate=0.01, n_iters=1000)
    p.fit(X_train, y_train)
    predictions = p.predict(X_test) 

    print("Perceptron classification accuracy", accuracy(y_test, predictions))
