import numpy as np

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.weights_input_hidden = np.random.randn(self.input_size, self.hidden_size)
        self.bias_hidden = np.zeros((1, self.hidden_size))
        self.weights_hidden_output = np.random.randn(self.hidden_size, self.output_size)
        self.bias_output = np.zeros((1, self.output_size))

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def forward(self, input_data):
        self.input_data = input_data
        self.hidden_layer_input = np.dot(input_data, self.weights_input_hidden) + self.bias_hidden
        self.hidden_layer_output = self.sigmoid(self.hidden_layer_input)
        self.output_layer_input = np.dot(self.hidden_layer_output, self.weights_hidden_output) + self.bias_output
        self.output_layer_output = self.sigmoid(self.output_layer_input)

    def backward(self, target):
        self.output_error = target - self.output_layer_output
        self.output_delta = self.output_error * self.sigmoid_derivative(self.output_layer_output)
        self.hidden_layer_error = self.output_delta.dot(self.weights_hidden_output.T)
        self.hidden_layer_delta = self.hidden_layer_error * self.sigmoid_derivative(self.hidden_layer_output)
        self.weights_hidden_output += self.hidden_layer_output.T.dot(self.output_delta)
        self.bias_output += np.sum(self.output_delta, axis=0, keepdims=True)
        self.weights_input_hidden += self.input_data.T.dot(self.hidden_layer_delta)
        self.bias_hidden += np.sum(self.hidden_layer_delta, axis=0, keepdims=True)

    def train(self, input_data, target, learning_rate):
        self.forward(input_data)
        self.backward(target)

    def predict(self, input_data):
        self.forward(input_data)
        return self.output_layer_output
input_size = 2
hidden_size = 4
output_size = 1

nn = NeuralNetwork(input_size, hidden_size, output_size)
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])
epochs = 10000
learning_rate = 0.1
for i in range(epochs):
    for j in range(len(X)):
        nn.train(X[j], y[j], learning_rate)
test_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
for data in test_data:
    prediction = nn.predict(data)
    print(f"Input: {data}, Prediction: {prediction}")
