import numpy as np
import random
import os

class Network(object):
    def __init__(self, sizes):
        self.num_layers = len(sizes)
        self.sizes = sizes
        
        # Initialize biases as a list
        self.biases = []
        for y in sizes[1:]:  # for every layer except the input layer
            b = np.random.randn(y, 1)  # column vector of biases
            self.biases.append(b)
        
        # Initialize weights as a list
        self.weights = []
        for x, y in zip(sizes[:-1], sizes[1:]):  # pairs of consecutive layers
            w = np.random.randn(y, x)  # weight matrix with shape (y, x)
            self.weights.append(w)

    def feedforward(self, a):
        #return output of network for 'a' --> vector input
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w,a) + b) #a' = sigmoid(w.a + b)
        return a

    def SGD(self, training_data, epochs, mini_batch_size, eta, test_data=None):
        """training_data is a list of tuples '(x,y)' representing 
        training inputs and desired output
        If 'test_data' is provided then the network will be evaluated against 
        test data after each epoch, and partial progress is printed out."""
        if test_data:
            n_test = len(test_data)
        n = len(training_data)
        for j in range(epochs):
            random.shuffle(training_data)
            mini_batches = [
                    training_data[k:k+mini_batch_size]
                    for k in range(0,n,mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)

            if test_data:
                print("Epoch {0}: {1} / {2}".format(j, self.evaluate(test_data), n_test))
            else:
                print("Epoch {0} complete".format(j))

    def update_mini_batch(self, mini_batch, eta):
        """Update the network's weights and biases by applying
        gradient descent using backpropagation to a single mini batch.
        The 'mini batch' is a list of tuples '(x,y)',
        and 'eta' is the learning rate."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        for x,y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]

            self.weights = [w-(eta/len(mini_batch))*nw
                			for w, nw in zip(self.weights, nabla_w)]
            self.biases = [b-(eta/len(mini_batch))*nb
                			for b, nb in zip(self.biases, nabla_b)]

    def backprop(self, x, y):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        activation = x
        activations = [x]
        zs = []

        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)

        delta = self.cost_derivative(activations[-1], y) * \
                sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())

        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (nabla_b, nabla_w)

    def evaluate(self, test_data):
        test_results = [(np.argmax(self.feedforward(x)), y)
            			for (x,y) in test_data]
        return sum(int(x==y) for (x,y) in test_results)

    def cost_derivative(self, ouput_activations, y):
        return (ouput_activations-y)
    
    def save(self, filename='model.npz'):
        path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'model'))
        os.makedirs(path, exist_ok=True)
        full_path = os.path.join(path, filename)
        np.savez_compressed(full_path, 
                            weights=np.array(self.weights, dtype=object), 
                            biases=np.array(self.biases, dtype=object))

    def load(self, filename='model.npz'):
        path = os.path.join('model', filename)
        data = np.load(path, allow_pickle=True)
        self.weights = data['weights']
        self.biases = data['biases']

def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))  #1/(1 + e^-z) for every element in z

def sigmoid_prime(z):
    return sigmoid(z) * (1-sigmoid(z)) #derivate of sigmoid function


