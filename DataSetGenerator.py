"""
    Algorithm used to generate a data set to train and test with.
"""
import random
import numpy as np

class DataSetGenerator():

    def __init__(self, start, end, number_of_points, ratio):
        """
        Instantiation of generator object. 

        :param start:   smallest possible value for data point <int>
        :param end:     largest possible value for data point <int>
        :param number_of_points: number of data points to generate <int>
        :param ratio: ratio of number of positive data points to negative data point <float> 

        Usage:: 

        >>> from Data_Set_Generator_10_19_2018 import DataSetGenerator
        >>> generator = DataSetGenerator(-10, 10, 15, 0.5)
        """
        self.start = start
        self.end = end
        self.number_of_points = number_of_points
        self.ratio = ratio

    def _random_data_point_generator(self):
        """
        Method used to generate a random data point. 
        This is a private method as indicated by the underscore in the beginning of the method name.

        Returns::
        :param data_point: [x1, x2] <list>

        Usage::

        >>> generator._random_data_point_generator()
        >>> [-2, 10] 
        """
        # Initialize variables
        data_point = []

        # Randomly generate two numbers and append to the data_point list to make it a point
        x1 = random.randint(self.start, self.end)
        x2 = random.randint(self.start, self.end)
        data_point.append(x1)
        data_point.append(x2)

        return data_point

    def _labeler(self, data_point):
        """
        Method used to label a data point as either positive or negative.
        This is a private method as indicated by the underscore in the beginning of the method name.

        Returns::
    
        1 : If x1 is greater than x2 <int>
        0 : If x1 is less than x2    <int>

        Usage::
    
        >>> data_point = [-2, 10]
        >>> generator._labeler(data_point)
        >>> 1
        """
        # Initialize variables 
        x1 = data_point[0]
        x2 = data_point[1]

        # If x1 is greater than x2, return negative label
        if x1 > x2:
            return 0

        # Else if x2 is greater than x1, return positive label    
        elif x1 < x2:
            return 1

        # Else, generate a new data_point and rerun method
        else:
            new_data_point = self._random_data_point_generator()
            self._labeler(new_data_point)

    def generate_data_set(self):
        """
        Method to generate a complete data set of a specified size and ratio with labels for each data point in the set.

        Returns::
        :param data_set: list of randomly generated data points <list>
        :param labels: list of labels for each data point <list>

        Usage::
        
        >>> data_set, labels = generator.generate_data_set()
        >>> data_set = [[1,2], [3, 4], [-1, -10]]
        >>> labels = [1, 1, -1]
        """
        # Initialize variables
        number_of_positive_labels = int(self.number_of_points*self.ratio)
        number_of_negative_labels = self.number_of_points - number_of_positive_labels
        positive_counter = 0
        negative_counter = 0
        counter = 0
        data_set = []
        labels = []

        # Fill up data_set list with random data points
        if self.number_of_points % 2 == 0:
            while counter < self.number_of_points:
                # Generate a random data point
                data_point = self._random_data_point_generator()

                # Make a label for the data_point
                label = self._labeler(data_point)

                # Check the label and increase counters
                if label == 1 and positive_counter < number_of_positive_labels:
                    data_set.append(data_point)
                    labels.append(label)
                    positive_counter += 1
                    counter += 1

                elif label == 0 and negative_counter < number_of_negative_labels:
                    data_set.append(data_point)
                    labels.append(label)
                    negative_counter += 1
                    counter += 1
        else:
            while counter < self.number_of_points:
                # Generate a random data point
                data_point = self._random_data_point_generator()

                # Make a label for the data_point
                label = self._labeler(data_point)

                # Check the label and increase counters
                if label == 1 and positive_counter < number_of_positive_labels:
                    data_set.append(data_point)
                    labels.append(label)
                    positive_counter += 1
                    counter += 1

                elif label == 0 and negative_counter < number_of_negative_labels:
                    data_set.append(data_point)
                    labels.append(label)
                    negative_counter += 1
                    counter += 1

        return np.array(data_set), np.array(labels)


                
    
