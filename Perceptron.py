import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
from matplotlib.animation import FuncAnimation
from matplotlib.animation import FFMpegWriter

class Perceptron():
    def __init__(self, data_set, labels, start, end, bias=1, learning_rate=0.001):
        """
        Initialization of a perceptron. 

        :param data_set: randomly generated data set<numpy array>
        :param labels:   labels associated with random data <numpy array>
        :param bias: bias term for perceptron line <int>
        :param learning_rate: rate at which the Perceptron learns <float>

        Usage::

        >>> from Perceptron_10_26_2018 import Perceptron
        >>> perceptron = Perceptron(data_set, labels)


        Notes::
        
        1. self.weights is assigned to be a numpy array with length data_set + 1 to accommodate for the bias term
        2. self.data_set is assigned to insert bias term in each data point of the data_set
        """
        self.labels = labels
        self.weights = np.zeros(len(data_set[0])+1)
        self.bias = bias
        self.learning_rate = learning_rate
        self.start = start
        self.end = end
        self.x_weights = [0]  #  This is to have the perceptron start off as a 
        self.y_weights = [1]  #  horizontal line so we can see it move and update
        self.bias_list = [0]  #  in the animation.
        self.data_set = np.insert(data_set, 0, self.bias, axis=1)

    def _activation(self, prediction):
        """
        Unit-step activation function. If Perceptron predicts a value greater than or equal to 0, returns 1. 
        Otherwise, function returns 0. 
        This is a private method as indicated by the underscore in the beginning of the method name.

        :param prediction: dot product of weight vector with an input vector <int>
        """
        return 1 if prediction >= 0 else 0
 
    def _predict(self, data_point):
        """
        This method predicts the output label of a data point within a data set by computing the dot 
        product of the weight vector with an input vector and then calling the activation method on 
        the dot product value.
        This is a private method as indicated by the underscore in the beginning of the method name. 

        Returns::
        
        :param prediction_label: either 1 or 0 <int>
        """
        prediction = self.weights.dot(data_point)
        prediction_label = self._activation(prediction)
        return prediction_label
 
    def train(self, epochs=100):
        """
        This method trains the perceptron by calling the prediction method on each data point within
        a data set. It then updates the weights by using the perceptron algorithm:
        weights = weights + learning_rate * (actual_label - predicted_label) * data_point

        :param epochs: number of times to run the perceptron algorithm <int>

        Usage::

        perceptron = Perceptron(data_set, labels, start, end)
        perceptron.train(100)

        Returns::

        :param self.weights: 1 by 3 numpy array with weights final weights for [bias, y_weight, x_weight]
        """
        for epoch in range(epochs):
            count = 0
            if epoch > 0:
                self.bias_list.append(self.weights[0]) # Append bias weight to list for every epoch
                self.x_weights.append(self.weights[1]) # Append x weight to list for every epoch
                self.y_weights.append(self.weights[2]) # Append y weight to list for every epoch
            for data_point in self.data_set:
                prediction_label = self._predict(data_point)
                true_label = self.labels[count]
                update = self.learning_rate * (true_label - prediction_label) * data_point
                self.weights = self.weights + update
                count += 1

        return self.weights

    def perceptron_animation(self):
        """
        Method for animating how a perceptron learns via a matplotlib plot.
        
        Usage::
        >>> perceptron = Perceptron(data_set, labels, start, end)
        >>> perceptron.train()
        >>> perceptron.perceptron_animation()

        NOTE: There was an error when running this: WindowsError: [Error 2] The system could not find the given data.
        1.  To resolve this go to C:/Users/username/AppData/Roaming/Python/Python37/site-packages/matplotlib/animation.py
        2.  Go down to class MovieWriter(AbstractMovieWriter):
        3.  def _run(self):
                self.proc = subprocess.Popen(...)
        4.  Change shell=False to shell=True

        References::
        >>> <https://stackoverflow.com/questions/9256829/how-can-i-save-animation-artist-animation/9281433#9281433>
        >>> <https://matplotlib.org/gallery/animation/simple_anim.html#sphx-glr-gallery-animation-simple-anim-py>
        """

        # Create a figure, one subplot, and legend --- subplot's name is 'ax'
        fig, ax = plt.subplots()
        perceptron_patch = mpatches.Patch(color='b', label='Perceptron')
        true_line_patch = mpatches.Patch(color='g', label='True Line')

        # Get all the x,y coordinates of the positive labeled data and negative labeled data in the data_set
        x_of_positive_points = [point[1] for point in self.data_set if point[2] > point[1]]
        y_of_positive_points = [point[2] for point in self.data_set if point[2] > point[1]]
        x_of_negative_points = [point[1] for point in self.data_set if point[2] < point[1]]
        y_of_negative_points = [point[2] for point in self.data_set if point[2] < point[1]]

        # Plot data_set
        ax.plot(x_of_positive_points, y_of_positive_points, 'ro', x_of_negative_points, y_of_negative_points, 'bo')

        # Plot actual equation 'y=x'
        x = np.linspace(self.start-1, self.end+1, len(self.x_weights))
        true_y = eval('x')
        ax.plot(x,true_y, 'g')

        # Plot initial starting point for the perceptron. This is a horizontal line. 
        line, = ax.plot(x, - (self.x_weights[0] / self.y_weights[0]) * x - (self.bias_list[0] / self.y_weights[0]))
        
        # Plot legend
        ax.legend(handles=[perceptron_patch, true_line_patch], loc='upper left')

        # Define the animation function
        def animate(i):
            line.set_ydata((-self.x_weights[i] / self.y_weights[i]) * x - (self.bias_list[i] / self.y_weights[i]))
            return line,

        # Animate the plot and show
        ani = animation.FuncAnimation(fig, animate)
        plt.show()
        
    def plot(self):
        """
        Method for simply plotting the end result of the perceptron's learning. 
        
        Usage::

        perceptron = Perceptron(data_set, labels, start, end)
        perceptron.train()
        perceptron.plot()
        """
        plt.clf()

        x_of_positive_points = [point[1] for point in self.data_set if point[2] > point[1]]
        y_of_positive_points = [point[2] for point in self.data_set if point[2] > point[1]]
        x_of_negative_points = [point[1] for point in self.data_set if point[2] < point[1]]
        y_of_negative_points = [point[2] for point in self.data_set if point[2] < point[1]]

        plt.plot(x_of_positive_points, y_of_positive_points, 'ro', x_of_negative_points, y_of_negative_points, 'bo')

        x = np.array(range(self.start-1, self.end+1))
        y = eval('-{w1}/{w2} * x - {b}/{w2}'.format(w1=self.weights[1], w2=self.weights[2], b=self.weights[0]))
        plt.plot(x,y, 'g')
        
        true_y = eval('x')
        plt.plot(x, true_y, 'r')

        ax = plt.gca()
        ax.set_ylim([0, self.end+1])
        ax.set_xlim([0, self.end+1])

        plt.show()
        




    