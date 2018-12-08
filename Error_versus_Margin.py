import numpy as np
import matplotlib.pyplot as plt
from DataSetGenerator import DataSetGenerator
from Perceptron import Perceptron
from numpy import linalg as LA
import statistics as stats

N = [2, 5, 10, 20, 50, 100]
error_means = []
error_standard_deviations = []
margin_means = []
margin_standard_deviations = []

for number_of_data_points in N:
    errors = []
    margins = []

    for i in range(100):
        # Initialize some variables for later 
        error_count = 0
        error = 0

        # Let's set some hyperparamaters for our data set generators
        number_of_data_points = number_of_data_points
        start = -100
        end = 100
        training_data_set_size = number_of_data_points
        test_data_set_size = 1000
        ratio = 0.5

        # Now to create the data set generator objects
        generator = DataSetGenerator(start, end, training_data_set_size, ratio)
        test_data_generator = DataSetGenerator(start, end, test_data_set_size, ratio)

        # Finally, create our test data_set and labels
        data_set, labels = generator.generate_data_set()
        test_data_set, test_labels = test_data_generator.generate_data_set()

        # Now let's create our perceptron object and train it with the training data
        perceptron = Perceptron(data_set, labels, start, end)
        perceptron_weights = perceptron.train()

        # Have perceptron line label all the test data
        perceptron_labels = []

        for test_data_point in test_data_set:
            test_data_point = np.insert(test_data_point, 0, 1, axis=0)
            perceptron_prediction = perceptron_weights.dot(test_data_point)

            if perceptron_prediction >= 0:
                perceptron_labels.append(1)
            else:
                perceptron_labels.append(0)

        # Now let's calculate error by counting how many labels the perceptron got wrong
        for i in range(len(perceptron_labels)):
            if perceptron_labels[i] != test_labels[i]:
                error_count = error_count + 1

        error = float(error_count*100)/len(perceptron_labels)
        errors.append(error)

        # Now let's find the closest margin
        # First, create a new list for margin labels because it needs 1 and -1 instead of 1 and 0
        margin_labels = []

        for label in labels: 
            if label != 0:
                margin_labels.append(1)
            else:
                margin_labels.append(-1)

        # Now list make a list for all margins in the data set 
        geometric_margins = []

        # To find the margin, we will need the unit vector of the perceptron weights
        magnitude = np.sqrt(perceptron_weights[1]*perceptron_weights[1] + perceptron_weights[2]*perceptron_weights[2])
        unit_vector = perceptron_weights/magnitude

        # Let's also modify the data set for calculating the margins
        margin_data_set = np.insert(data_set, 0, 1, axis=1)

        # Calculate the geometric margin for each data point in the data set
        for i in range(len(margin_data_set)):
            geometric_margins.append(margin_labels[i]*(margin_data_set[i].dot(unit_vector)))

        # The best geometric margin is the smallest of the geometric margin
        best_geometric_margin = min(geometric_margins)
        margins.append(best_geometric_margin)

    error_means.append(stats.mean(errors))
    error_standard_deviations.append(stats.stdev(errors))


    margin_means.append(stats.mean(margins))
    margin_standard_deviations.append(stats.stdev(margins))

print(margin_means)
print(error_means)
plt.figure(1)
plt.subplots_adjust(hspace=0.5)

plt.subplot(211)
plt.xlim((N[0]-10, N[-1]+10))
plt.xlabel('Number of Data Points')
plt.ylabel('Error [%]')
plt.errorbar(N, error_means, yerr=error_standard_deviations)
plt.title('Error Percentage versus Number of Data Points')

plt.subplot(212)
plt.xlim(min(margin_means)-2, max(margin_means)+2)
plt.xlabel('Mean of Margins')
plt.ylabel('Mean of Errors')
plt.title('Means of Error Percentage versus Means of Largest Margins')
plt.plot(margin_means, error_means, 'r')
plt.show()