import numpy as np
import matplotlib.pyplot as plt
from DataSetGenerator import DataSetGenerator
from Perceptron import Perceptron

# Hyperparamters
start = -10000
end = 10000
number_of_data_points = 100
ratio = 0.90


# Make a data set generator object
generator = DataSetGenerator(start, end, number_of_data_points, ratio)

# Generate a data set and labels
data_set, labels = generator.generate_data_set()

# Make a perceptron object
perceptron = Perceptron(data_set, labels, start, end)

# Train the perceptron
perceptron.train()

# Delete this after
#perceptron.minimum_distance()

# Plot animation
perceptron.perceptron_animation()

