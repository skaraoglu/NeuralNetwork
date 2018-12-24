"""########################################################

Building Simple Neural Network using numpy.

This code allows user to define the number of layers and
the number of nodes in each layer.

For the hidden layers, currently available functions are
tanh, relu and softmax.

Selim Karaoglu, 2018.

########################################################"""

#Necessaray Libraries
import numpy as np
import time
import collections as col
import matplotlib.pyplot as plt

"""Activation functions and derivative functions are defined here
@x : Data
@f : Activation function name
"""
def activation_f(x, f):
    if f == "relu":
        return x * (x > 0)
    elif f == "tanh":
        return np.tanh(x)
    elif f == "softmax":
        return np.exp(x) / np.sum(np.exp(x), axis = 1).reshape((-1,1))

def derivative_f(x, f):
    if f == "relu":
        return 1. * (x > 0)
    elif f == "tanh":
        return 1 - np.tanh(x) ** 2
    elif f == 'softmax':
        tmp = x.reshape(-1, 1)
        return np.diagflat(tmp) - np.dot(tmp, tmp.T)

"""This function is used to build the Neural Network model
Takes a list of layers as an input and returns layers.
@model : Model structure as a list of layers.
"""
def BuildModel(model):
    layers = []
    for f, num_input_layers, num_output_layers in model:
        l = layer(f, num_input_layers, num_output_layers)
        layers.append(l)
    return np.asarray(layers)

"""This function is used to build the layers of the Neural
Network. Takes activation function name, number of input
layers and number of output layers as an input, returns a
list that contains zero filled array of weights and bias,
and the activation function name.
@f                 : Activation function name
@num_input_layers  : Number of input layers
@num_output_layers : Number of output layers
"""
def layer(f, num_input_layers, num_output_layers):
    w = ((2 * np.random.random((num_input_layers, num_output_layers)) - 1) / num_input_layers)
    b = np.zeros(num_output_layers)
    f = f
    return [w, b, f]

"""Forward propagation function
This function forward propagates layers using activation
function of the layer.
@x      : Data
@layers : Layers as a list
Returns activations list to feed to the next layer
"""
def forward(x, layers):
    activations = [x]
    for i in range(len(layers)):
        x = activation_f((np.dot(x, layers[i][0]) + layers[i][1]), layers[i][2])
        activations.append(x)
    return activations

"""Prediction function
@X      : Data
@layers : Layers as a list
Returns predictions as a list
"""
def Predict(X, layers):
    activations = forward(X, layers)
    predictions = activations[-1]
    return predictions

#Pretty obvoius isn't it?
def forwardPredict(X, layers):
    activations = forward(X, layers)
    predictions = activations[-1]
    return activations, predictions

"""Backward Propagation function
This function back propagates layers using gradients.
@activations   : Activation matrix from last forward propagation
@t             : Output labels
@learning_rate : Rate of adjustments on weights and bias
@layers        : Layers as a list
Returns cost
"""
def backward(activations, t, learning_rate, layers):
    cost = 0
    grad = None
    delt = col.deque()
    
    for i in reversed(range(len(layers))):
        y = activations[i+1]
        a = activations[i]

        if grad is None:
            cost = crossEntropyLoss(y, t, t.shape[0])
            grad = (y - t)
        else:
            dLdh = grad.dot(layers[i+1][0].T)
            grad = np.multiply(dLdh, derivative_f(y, layers[i][2]))

        grad_w = 1/t.shape[0] * (a.T).dot(grad)
        grad_b = 1/t.shape[0] * np.sum(grad, axis=0)        
        delt.appendleft((grad_w, grad_b))

    for j in reversed(range(len(layers))):
        grad_w, grad_b = delt[j]
        layers[j][0] -= grad_w * learning_rate
        layers[j][1] -= grad_b * learning_rate

    return cost

"""This function calculates the accuracy by using argmax
argument on truth and predicted labels and comparing both.
@truth : One Hot Vector encoded labels
@pred  : Predictions matrix
Returns accuracy
"""
def CalculateAccuracy(truth, pred):
    if len(truth) != len(pred):
        raise ValueError("Error: Array size mismatch!")

    true = 0
    x_ = [np.argmax(i) for i in truth]
    y_ = [np.argmax(j) for j in pred]

    for x, y in zip(x_, y_):
        if x == y:
            true += 1

    return true / len(truth)
    
"""This function is used to train the Neural Network
@layers        : Layers as a list
@X             : Training Data
@y             : Training Labels
@num_epochs    : Number of epochs (default = 100)
@batch_s       : Batch size (default = 50)
@learning_rate : Rate of adjustments on weights and bias (default =  0.01)
Returns Validation cost and Prediction scores.
"""
def Train(layers, X, y, num_epochs=100, batch_s=50, learning_rate=.01):
    print("Training started, may take few minutes.")
    s = time.time()
    _train = list(zip(X, y))
    v_cost = []
    p_score = []

    for i in range(num_epochs):
        np.random.shuffle(_train)
        mini_batch = [_train[k:k+batch_s] for k in range(0, len(_train), batch_s)]

        b_cost = []
        b_score  = []
        for _b in mini_batch:
            X = np.asarray([x[0] for x in _b])
            y = np.asarray([x[1] for x in _b])

            activations, predictions = forwardPredict(X, layers)

            accuracy = CalculateAccuracy(y, predictions)
            b_score.append(accuracy)

            cost = backward(activations, y, learning_rate, layers)
            b_cost.append(cost)

        v_cost.append(np.mean(b_cost))
        p_score.append(np.mean(b_score))
    print("Training completed in %s seconds." % str(time.time()-s))
    return v_cost, p_score

"""Cross Entropy Loss function.
Used to calculate cross entropy loss un backward propagation.
"""
def crossEntropyLoss(y, _t, _m):
    return - np.multiply(_t, np.log(y)).sum() / _m

"""Function to encode labels as one hot vectors."""
def OneHot(X):
    z = np.zeros((len(X), 10))
    z[np.arange(len(z)), X] += 1
    return z

def PlotResults(score, accuracy, cost):
    print("Training Accuracy = %s" % str(score[-1] * 100))
    print("Test Accuracy = %s" % str(accuracy * 100))
    fig, axes = plt.subplots(2,1, figsize=(20,20))
    axes[0].set_title("Validation costs")
    axes[1].set_title("Training Accuracy")
    for x, y in enumerate(cost):
        axes[0].scatter(x, y, color='purple', alpha=0.75)
    for x, y in enumerate(score):
        axes[1].scatter(x, y, color='orange', alpha=0.75)
    plt.tight_layout()
    plt.show()
    