import numpy as np

def stepFunction(val):
    """Step function would be use while training neurons and it is basically a hard limitor function which generate a output one if netsum value from Perceptron class fit funtion is greater than threshold. Thresold value by default we are setting to 0.5"""
    threshold = 0.5
    if val > threshold:
        return 1
    else:
        return 0

def perceptron(X, y, epochs):
    """This is the Function for perceptron in which we have three parameters that are X which is basically 2D input features  which we are giving to our perceptron model for eg OR,AND gates input or any linear separable inputs with target variable which is an numpy array and number of Epochs to consider. """
    bias = 0
    learningRate = 0.2
    weights = np.zeros(X.shape[1])
    print(f"Initial weights are {weights}")
    for i in range(epochs):
        for index in range(len(X)): 
            # print(X[index])
            netsum = np.dot(X[index], weights) + bias
            y_predicted = stepFunction(netsum)
            error = y[index] - y_predicted
            if error != 0:
                weights += learningRate * error * X[index]
                bias += learningRate * error
            
        print(f"Iteration {i+1}: Weights = {weights}, Bias = {bias}")
    
    return weights, bias

def predict(X, weights, bias):
    netsum = np.dot(X, weights) + bias
    return stepFunction(netsum)

if __name__ == "__main__":
    # OR Gate
    a = np.array([[0,0], [0, 1], [1, 0], [1, 1]]) 
    b = np.array([0, 1, 1, 1])  # Target labels
    
    # We are training the perceptron model getting final weights and bias :)
    w, bias = perceptron(a, b, 10) 

    test_input = np.array([2, 1])
    print(f"Prediction for {test_input}: {predict(test_input, w, bias)}")
