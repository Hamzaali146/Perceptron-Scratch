import numpy as np 

def stepFunction(val):
    threshold = 0.5
    if val > threshold:
        return 1
    else:
        return 0

def perceptron(X,y,iter):
    bias = 0
    learningRate=0.2
    weights = np.zeros(len(X))
    # print(weights)
    Flag = True
    count = 0
    for i in range(iter):
        netsum = np.dot(X,weights)+ bias
        y_predicted = stepFunction(netsum)
        error = y[count] - y_predicted
        # print(error)
        if error != 0:
            weights+= learningRate*error*X
            bias = learningRate*error
    return weights,bias
def predict(X,weights,bias):
    netsum = np.dot(X,weights)+ bias
    y_predicted = stepFunction(netsum)
    return y_predicted
a=np.array([1,2,3,4,5,6])
b=np.array([1,1,0,1,0,0])
c=np.array([2,3,4,5,6,8])
w,bias = perceptron(a,b,5)
print(w,bias)
print(predict(c,w,bias))

# print(a[0])
# c = np.zeros(6)
# print(stepFunction(np.dot(a,b)))
# print(np.random(6))

if __name__ == "__main__":
    a=np.array([1,2,3,4,5,6])
    b=np.array([1,1,0,1,0,0])
    c=np.array([2,3,4,5,6,8])
    w,bias = perceptron(a,b,5)
    print(w,bias)
    print(predict(c,w,bias))