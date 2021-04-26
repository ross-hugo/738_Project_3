import random
import numpy as np
from tqdm.notebook import tqdm
import pandas as pd

class NN():
    """
    Setting up the NN by setting up the weights (with random values)
    """
    def __init__(self, input_nodes, output_nodes, hidden_layers):
        self.input_nodes = input_nodes
        self.output_nodes = output_nodes
        self.hidden_layers = hidden_layers
        self.weights = self.initializeWeights()

        self.activations = [None] * (len(hidden_layers) + 1)
        return
    
    """
    Top level function for training the model. For each iteration it
    runs the feed forward of calculating the function that is a mapping
    from X to y. The errors are then computed and then the weights are
    then calibrated through back propegation.
    """
    def train(self, X, y, epochs=10, batch_size=32, learning_step=1):
        X = np.array(X)
        X = X / np.amax(X)

        for _ in tqdm(range(epochs)):
			# get a random batch
            inputs, temp_outputs = self.get_batch(X, y, batch_size)

            #Use a one hot notation for all inputs and outputs. 
            #Makes things easier especially with the loss function used SGD
            outputs = []
            for i in range(batch_size):
                a = np.zeros(self.output_nodes, dtype=np.int8)
                a[temp_outputs[i]] = 1
                outputs.append(a)

            #where the real training happens woop
            for i in range(len(inputs)):
                self.feed_forward(inputs[i])
                self.back_prop(inputs[i], outputs[i], learning_step)

        correct = 0
        verify_inputs, verify_outputs = self.get_batch(X, y, batch_size)
        for i in range(len(verify_inputs)):
            prediction = self.predict(verify_inputs[i])
            if prediction == verify_outputs[i]:
                correct += 1
        print('Train accuracy:', round(correct / len(verify_inputs), 6))

    """
    Gets randomiized weights
    """
    def initializeWeights(self):
        ret  = []
        for i in range(len(self.hidden_layers) + 1):
            if i == 0:
                ret.append(np.random.randn(self.input_nodes, self.hidden_layers[0]))
            elif i != len(self.hidden_layers):
                ret.append(np.random.randn(self.hidden_layers[i - 1], self.hidden_layers[i]))
            else:
                ret.append(np.random.randn(self.hidden_layers[-1], self.output_nodes))
        return np.array(ret)

    """
    This produces ouput from a given input X
    """
    def feed_forward(self, X):
        def sigmoid(z):
            return 1/(1 + np.exp(-z))

        self.activations[0] = sigmoid(np.dot(X, self.weights[0]))
        for i in range(len(self.hidden_layers) - 1):
            self.activations[i + 1] = sigmoid(np.dot(self.activations[i], self.weights[i + 1]))
        self.activations[-1] = sigmoid(np.dot(self.activations[-2], self.weights[-1]))

    """
    Where the weights are changed. where the real training portion happens in reducing the error 
    between the expected output and the realized output from the model.
    """
    def back_prop(self, X, y, learning_rate):
        def sig_deriv(z):
            return z * (1-z)

        # record errors in each layer, starting with the last layer. Hence ~back~ propegation 
        errors = []
        errors.append((y - self.activations[-1]) * sig_deriv(self.activations[-1]))
        for i in range(len(self.hidden_layers), 0, -1):
            errors.append((np.dot(errors[-1], self.weights[i].T)) * sig_deriv(self.activations[i - 1]))
        errors.reverse()

        #calibrate all weights in the model
        for i in range(len(self.weights)):

            layer = X if i == 0 else self.activations[i-1]

			# get corresponding error
            error = errors[i]

            #make sure that the mat mult works
            layer.shape = (len(layer), 1)
            error.shape = (1, len(error))

            self.weights[i] += learning_rate * np.dot(layer, error)

    """
    selects random batch or amount of data for training
    """
    def get_batch(self, X,  y, batch_size):
        ins, outs = [], []

        rand_samples = random.sample(range(len(X)), batch_size)

        for index in rand_samples:
            ins.append(X[index])
            outs.append(y[index])

        return np.array(ins), outs 
    
    """
    used in testing mainly. 
    Predict the output from the model from one piece of data
    """
    def predict(self, X):
        self.feed_forward(X)
        return np.argmax(self.activations[-1])

    """
    Verifying the accuracy of the model. USE DIFFERENT data than the data used in training
    This way you get a better picture of how well the model performs
    """
    def test(self, X, y):
        correct = 0

		# normalize inputs
        X = np.array(X)
        X = X / np.amax(X)

        incorrect = []

		# classify test data
        for i in tqdm(range(len(X))):
            prediction = self.predict(X[i])
            if prediction == y[i]:
                correct += 1

        print('Test accuracy:', round(correct / len(X), 6))
