import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import statistics as stats
import random as rand
from DataSetGenerator import DataSetGenerator
from Perceptron import Perceptron

N = [2, 5, 10, 20, 50, 100]
means = []
standard_deviations = []

for number_of_data_points in N:
    weights = []
    errors = []

    # Hyperparamters
    start = -100
    end = 100
    number_of_data_points = number_of_data_points
    ratio = 0.5

    for i in range(100):
        # Make a data set generator object
        generator = DataSetGenerator(start, end, number_of_data_points, ratio)

        # Generate a data set and labels
        data_set, labels = generator.generate_data_set()

        # Make a perceptron object
        perceptron = Perceptron(data_set, labels, start, end)

        # Train the perceptron
        weights.append(perceptron.train())
    
    for weight in weights:
        # Calculate error by taking dot product of weights with actual line
        actual_line = np.array([0, 1, 1])
        magnitude = np.sqrt(actual_line[1]*actual_line[1] + actual_line[2]*actual_line[2])
        unit_vector = actual_line/magnitude
        error = abs(weight.dot(unit_vector))
        errors.append(error)

    means.append(stats.mean(errors))
    standard_deviations.append(stats.stdev(errors))

# Adjust standard deviation such that the error does not dip below zero
for i in range(len(means)):
    if means[i] - standard_deviations[i] < 0:
        standard_deviations[i] = means[i]-0.0001

max_errors = []
for i in range(len(means)):
    max_errors.append(means[i] + standard_deviations[i])

max_error = max(max_errors)

print("Means: ")
print(means)
print("Standard Deviations: ")
print(standard_deviations)    

plt.subplot()
plt.xlim((N[0]-10, N[-1]+10))
plt.ylim(-0.01, max_error+0.01)
plt.xlabel('Number of Data Points')
plt.ylabel('Deviation')
plt.errorbar(N, means, yerr=standard_deviations, lolims=0)
plt.title('Deviation versus Number of Data Points')
plt.show()