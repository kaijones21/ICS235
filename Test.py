import numpy as np
import matplotlib.pyplot as plt
import statistics as stats
from DataSetGenerator import DataSetGenerator
from Perceptron import Perceptron

N = [2, 5, 10, 20, 50, 100]
error_means = []
error_standard_deviations = []
margin_means = []
margin_standard_deviations = []

for number_of_data_points in N:
    errors = []
    margins = []

    # Run simulation 100 times for each number of data points
    for i in range(100):    
        # Initialize variables
        perceptron_labels = []
        error_count = 0
        error = 0
    
        # Hyperparamters
        start = -100
        end = 100
        training_data_set_size = number_of_data_points
        ratio = 0.50
        test_data_set_size = 50
    
        # Make a data set generator object
        generator = DataSetGenerator(start, end, training_data_set_size, ratio)
    
        # Generate a training data set and labels
        data_set, labels = generator.generate_data_set()
    
        # Make a perceptron object
        perceptron = Perceptron(data_set, labels, start, end)
    
        # Train the perceptron
        perceptron_weights = perceptron.train()
    
        # Make a new data set generator object for generating test data 
        test_data_generator = DataSetGenerator(start, end, test_data_set_size, ratio)
    
        # Generate a test data set and labels
        test_data_set, test_labels = test_data_generator.generate_data_set()

        #print('Test Labels: ')
        #print(test_labels)
        #print('\n')
    
        # Have perceptron line label all data_points in the test_data_set and save into a list
        for test_data_point in test_data_set:
            perceptron_prediction = -perceptron_weights[1]/perceptron_weights[2] * test_data_point[0] - perceptron_weights[0]/perceptron_weights[2]
    
            if perceptron_prediction > test_data_point[0]:
                perceptron_labels.append(1)
    
            elif perceptron_prediction < test_data_point[0]:
                perceptron_labels.append(0)

        #print('Perceptron Labels: ')
        #print(perceptron_labels)
        #print('\n')
    
        # Count how many times the perceptron labeled a data_point wrong and calculate error by
        # dividng count by size of label set
        for i in range(len(perceptron_labels)):
            if perceptron_labels[i] != test_labels[i]:
                error_count = error_count + 1
    
        error = float(error_count)*100/len(test_labels)
        errors.append(error)
        #print("Error Percentage: {error} %".format(error=error))
    
        # For the purpose of finding the optimal margin, we will need to change all the zeros of the labels to -1
        for i in range(len(labels)):
            if labels[i] == 0:
                labels[i] = -1
    
        # List that will contain all the margins
        geometric_margins = []
    
        #print('Weight: ')
        #print(perceptron_weights)
        #print('\n')
    
        # Calculate the magnitude of the weight vector (excluding bias)
        magnitude = np.sqrt(perceptron_weights[1]*perceptron_weights[1] + perceptron_weights[2]*perceptron_weights[2])
    
        #print('Magnitude: ')
        #print(magnitude)
        #print('\n')
    
        # Find the unit vector by dividing perceptron weights (including bias) by the magnitude fo the weights
        unit_vector = perceptron_weights/magnitude
    
        #print('Unit Vector: ')
        #print(unit_vector)
        #print('\n')
    
        # Insert a 1 in the beginning of the data_set to align dimensions of the data_points with unit_vector 
        data_set = np.insert(data_set, 0, 1, axis=1)
    
        # Find margin for each data_point
        for i in range(len(data_set)):
            geometric_margins.append(labels[i]*(unit_vector.dot(data_set[i])))
    
        #print('Geometric Margins: ')
        #print(geometric_margins)
        #print('\n')
    
        # Find the optimal margin 
        best_geometric_margin = min(geometric_margins)
        margins.append(best_geometric_margin)

        #print('Best Geometric Margin: ')
        #print(best_geometric_margin)
        #print('\n')

    error_means.append(stats.mean(errors))
    error_standard_deviations.append(stats.stdev(errors))

    margin_means.append(stats.mean(margins))
    margin_standard_deviations.append(stats.stdev(margins))

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
plt.plot(margin_means, error_means, 'r--')
plt.show()