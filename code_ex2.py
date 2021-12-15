import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

MIN_RANGE = 0
MAX_RANGE = 1


def get_uniform_data(num_of_points, range_x, range_y):
    data_x = np.random.uniform(range_x[MIN_RANGE], range_x[MAX_RANGE], (1, num_of_points))
    data_y = np.random.uniform(range_y[MIN_RANGE], range_y[MAX_RANGE], (1, num_of_points))
    return data_x, data_y


def plot_data(x_array, y_array):
    plt.scatter(x_array, y_array)
    plt.show()


if __name__ == '__main__':
    x, y = get_uniform_data(5)
    plot_data(x, y)
