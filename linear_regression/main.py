import random
import numpy as np
from numpy.core.fromnumeric import argmax
# from network import sigmoid
from src import mnist_loader


#this is my personal hand-coded implementation of a neural-net for digit classifying

def apply_sigmoid_function(x):
    return 1.0/(1.0+np.exp(-x))

def apply_sigmoid_derivative(x):
    return apply_sigmoid_function(x)*(1-apply_sigmoid_function(x))

def apply_cost_derivative(output_activations, y):
    return (output_activations - y)

def back_propogate(error, weights, biases):
    pass


def feedforward(a):
    """Return the output of the network if ``a`` is input."""
    for b, w in zip(biases, weights):
        a = apply_sigmoid_function(np.dot(w, a)+b)
    return a

def evaluate(test_data):
    """Return the number of test inputs for which the neural
    network outputs the correct result. Note that the neural
    network's output is assumed to be the index of whichever
    neuron in the final layer has the highest activation."""
    test_results = [(np.argmax(feedforward(x)), y)
                    for (x, y) in test_data]
    test_results = []
    for x, y in test_data:
        model_output_vector = feedforward(x)
        max = np.argmax(model_output_vector)
        test_results.append((max, y))

    summ = 0
    for x, y in test_results:
        if x == y:
            summ+=1
    return summ

layer_sizes = [784, 30, 10]
biases = [np.random.randn(layer_size, 1) for layer_size in layer_sizes[1:]]
weights = [np.random.randn(adjacent_layer_size, current_layer_size) for current_layer_size, adjacent_layer_size in zip(layer_sizes[:-1], layer_sizes[1:])]
learn_rate = 3.0
training_data, validation_data, test_data = mnist_loader.load_data_wrapper('C:/Users/crazy/Documents/Dev/neural-networks-and-deep-learning/data/mnist.pkl.gz')
test_data = list(test_data)
training_data = list(training_data)
mini_batch_size = 10
epochs = 30
n = len(training_data)

for epoch in range(epochs):
    random.shuffle(training_data)
    mini_batches = [
        training_data[k:k+mini_batch_size] for k in range(0, n, mini_batch_size)
    ]

    for mini_batch in mini_batches:
        nabla_b = [np.zeros(b.shape) for b in biases]
        nabla_w = [np.zeros(w.shape) for w in weights]

        for input_vector, correct_output_vector in mini_batch:

            ############## Feed Forward #################
            activation_vector = input_vector
            activation_vectors = [activation_vector]
            weighted_sum_vectors = []
            for weight_matrix, bias_vector in zip(weights, biases):
                weighted_sum_vector = np.dot(weight_matrix, activation_vector) + bias_vector
                weighted_sum_vectors.append(weighted_sum_vector)
                activation_vector = apply_sigmoid_function(weighted_sum_vector)
                activation_vectors.append(activation_vector)
            ##############################################
            delta_nabla_b = [np.zeros(b.shape) for b in biases]
            delta_nabla_w = [np.zeros(w.shape) for w in weights]

            network_output_vector = activation_vectors[-1]
            derivative_of_cost_with_respect_to_output_activations = apply_cost_derivative(network_output_vector, correct_output_vector)
            derivative_of_output_activations_with_respect_to_weighted_sums = apply_sigmoid_derivative(weighted_sum_vectors[-1])
            delta = derivative_of_cost_with_respect_to_output_activations * derivative_of_output_activations_with_respect_to_weighted_sums
            delta_nabla_b[-1] = delta
            delta_nabla_w[-1] = np.dot(delta, activation_vectors[-2].transpose())

            for reverse_layer_index in range(2, len(layer_sizes)): # iterating over hidden layers in reverse starting from last hidden layer
                layer_weighted_sum_vector = weighted_sum_vectors[-reverse_layer_index]
                derivative_of_layer_activations_with_respect_to_layer_weighted_sums = apply_sigmoid_derivative(layer_weighted_sum_vector)
                weight_matrix = weights[-reverse_layer_index+1]
                transpose_weight_matrix = weight_matrix.transpose()
                delta = np.dot(transpose_weight_matrix, delta) * derivative_of_layer_activations_with_respect_to_layer_weighted_sums
                delta_nabla_b[-reverse_layer_index] = delta
                delta_nabla_w[-reverse_layer_index] = np.dot(delta, activation_vectors[-reverse_layer_index-1].transpose())
            
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        
        new_weights = []
        for weight_matrix, nabla_weight_matrix in zip(weights, nabla_w):
            weight_matrix_nudge = nabla_weight_matrix*learn_rate
            averaged_weight_matrix_nudge = weight_matrix_nudge/len(mini_batch)
            nudged_weight_matrix = weight_matrix - averaged_weight_matrix_nudge
            new_weights.append(nudged_weight_matrix)
        weights = new_weights


        new_biases = []
        for bias_vector, nabla_bias_vector in zip(biases, nabla_b):
            bias_vector_nudge = nabla_bias_vector*learn_rate
            averaged_bias_vector_nudge = bias_vector_nudge/len(mini_batch)
            nudged_bias_vector = bias_vector - averaged_bias_vector_nudge
            new_biases.append(nudged_bias_vector)
        biases = new_biases
        
    print("Epoch {} : {} / {}".format(epoch, evaluate(test_data), len(test_data)))

