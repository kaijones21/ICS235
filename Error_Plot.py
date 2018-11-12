import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import statistics as stats
from DataSetGenerator import DataSetGenerator
from Perceptron import Perceptron

N = [1, 5, 10, 20, 50, 100]
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
        # Calculate error by angle deviation
        theta1 = np.degrees(np.arctan(1)) # Reference angle for y=x
        theta2 = np.degrees(np.arctan(-weight[1]/weight[2])) # Angle of perceptron line
        error = abs(theta1-theta2)
        errors.append(error)

    means.append(stats.mean(errors))
    standard_deviations.append(stats.stdev(errors))

print("Means: ")
print(means)
print("Standard Deviations: ")
print(standard_deviations)    

plt.subplot()
plt.xlim((N[0]-10, N[-1]+10))
plt.xlabel('Number of Data Points')
plt.ylabel('Delta Angle [degrees]')
plt.errorbar(N, means, yerr=standard_deviations)
plt.title('Angle Deviation versus Number of Data Points')
plt.show()