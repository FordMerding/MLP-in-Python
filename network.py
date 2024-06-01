import random

import numpy as np
import dataloader as dl
import pickle

class Network:
    def __init__(self, sizes):
        self.num_layers = len(sizes)
        self.biases = [np.random.randn(b) for b in sizes[1:]]
        self.weights = [np.random.randn(n, p) for p, n in zip(sizes[:-1], sizes[1:])]
        self.sizes = sizes

    def forward(self, a):
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a) + b)
        return a
    
    def SGD(self, train_data, epochs, batch_size, eta):
        n = len(train_data)
        for epoch in range(1, epochs+1):
            random.shuffle(train_data)
            batches = [train_data[k:k+batch_size] for k in range(0, n, batch_size)]
            for batch in batches:
                self.UMB(batch, len(batch), eta)
            print(f"Epoch {epoch} is complete.")
        print(f"Training complete")
    
    def UMB(self, batch, size, eta):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for i in batch:
            deltanb, deltanw = self.backprop(i)
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, deltanb)]
            nabla_w = [nb + dnw for nb, dnw in zip(nabla_w, deltanw)]
        
        self.biases = [b - (eta * nb)/size for b, nb in zip(self.biases, nabla_b)]
        self.weights = [w - (eta * nw)/size for w, nw in zip(self.weights, nabla_w)]

    def backprop(self, train_data):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        x = train_data[0]
        activations = [x]
        zs = []

        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, x) + b
            zs.append(z)
            x = sigmoid(z)

            activations.append(x)
        delta = self.cost_func(activations[-1], train_data[1]) * sigmoid_prime(zs[-1])

        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(np.reshape(delta, (-1, 1)), np.reshape(activations[-2], (-1, 1)).transpose())

        for l in range(2, self.num_layers):
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sigmoid_prime(zs[-l])
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(np.reshape(delta, (-1, 1)), np.reshape(activations[-l-1], (-1, 1)).transpose())
        return nabla_b, nabla_w
    
    def evaluate(self, test_data):
        result = []
        total = 0
        for data in test_data:
            output = np.argmax(self.forward(data[0]))
            result.append((output, data[1]))
            if(output == data[1]):
                total+=1
        return (total/len(test_data), result)
    
    def cost_func(self, a, y):
        return (a - y)
    
    def save(self, name):
        with open(f'{name}.pkl', 'wb') as file:
            pickle.dump([self.weights, self.biases], file)
    
    def load(self, name):
        with open(f'{name}.pkl', 'rb') as file:
            loaded = pickle.load(file)
            self.weights = loaded[0]
            self.biases = loaded[1]
def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))
def sigmoid_prime(z):
    sig = sigmoid(z)
    return sig*(1.0-sig)
def vectorize_label(label):
    y = np.zeros(10)
    y[label] = 1.0
    return y
if(__name__ == "__main__"):
    layers = [28 * 28, 16, 16, 10]
    model = Network(layers)


    train_data = dl.load_data('./data/train-images-idx3-ubyte', 60000)
    train_labels = dl.load_labels('./data/train-labels-idx1-ubyte', 60000)

    test_data = dl.load_data('./data/t10k-images-idx3-ubyte', 10000)
    test_labels = dl.load_labels('./data/t10k-labels-idx1-ubyte', 10000)

    train_tuple = [(x.ravel(), vectorize_label(y)) for x, y in zip(train_data, train_labels)]
    test_tuple = [(x.ravel(), y) for x, y in zip(test_data, test_labels)]

    model.SGD(train_tuple, 40, 32, 0.1)

    accuracy, _ = model.evaluate(test_tuple)

    print(f"Accuracy: {accuracy*100}%")

    ok = input("Would you prefer to save it? ")
    if(ok == "Yes"):
        model.save("model")