import numpy as np
import matplotlib.pyplot as plt
from DataSetGenerator import DataSetGenerator
from Perceptron import Perceptron
from numpy import linalg as LA

# Initialize variables
perceptron_labels = []
error_count = 0
error = 0

# Hyperparamters
start = -100
end = 100
training_data_set_size = 100
ratio = 0.5
test_data_set_size = 100

# Make a data set generator object
generator = DataSetGenerator(start, end, training_data_set_size, ratio)

# Generate a training data set and labels
data_set, labels = generator.generate_data_set()

# Make a perceptron object
perceptron = Perceptron(data_set, labels, start, end)

# Train the perceptron
perceptron_weights = perceptron.train()

# For the purpose of finding the optimal margin, we will need to change all the zeros of the labels to -1
for i in range(len(labels)):
    if labels[i] == 0:
        labels[i] = -1

# List that will contain all the margins
geometric_margins = []

print('Weight: ')
print(perceptron_weights)
print('\n')

# Calculate the magnitude of the weight vector (excluding bias)
magnitude = np.sqrt(perceptron_weights[1]*perceptron_weights[1] + perceptron_weights[2]*perceptron_weights[2])

print('Magnitude: ')
print(magnitude)
print('\n')

# Find the unit vector by dividing perceptron weights (including bias) by the magnitude fo the weights
unit_vector = perceptron_weights/magnitude

print('Unit Vector: ')
print(unit_vector)
print('\n')

# Insert a 1 in the beginning of the data_set to align dimensions of the data_points with unit_vector 
data_set = np.insert(data_set, 0, 1, axis=1)

# Find margin for each data_point
for i in range(len(data_set)):
    geometric_margins.append(labels[i]*(unit_vector.dot(data_set[i])))

print('Geometric Margins: ')
print(geometric_margins)
print('\n')

# Find the optimal margin 
best_geometric_margin = min(geometric_margins)

print('Best Geometric Margin: ')
print(best_geometric_margin)
print('\n')

# Separate Positive and Negative Data 
positive_data = []
negative_data = []

for data_point in data_set:
    if data_point[2] > data_point[1]:
        positive_data.append(data_point)
    else:
        negative_data.append(data_point)

distance_of_positive_data = []
distance_of_negative_data = []

for positive_data_point in positive_data:
    distance_of_positive_data.append(unit_vector.dot(positive_data_point))

for negative_data_point in negative_data:
    distance_of_negative_data.append(abs(unit_vector.dot(negative_data_point)))

# Find the closest positive and negative data points
positive_data_point_and_distances = list(zip(positive_data, distance_of_positive_data))
negative_data_point_and_distances = list(zip(negative_data, distance_of_negative_data))


positive_support_vector = min(positive_data_point_and_distances, key = lambda t: t[1])
negative_support_vector = min(negative_data_point_and_distances, key = lambda t: t[1])

print('Positive Data Point and Distance: ')
print(positive_support_vector)
print('\n')

print('Negative Data Point and Distance: ')
print(negative_support_vector)
print('\n')

support_vectors = []

support_vectors.append(positive_support_vector[0])
support_vectors.append(negative_support_vector[0])

print('Support Vectors: ')
print(support_vectors)
print('\n')

