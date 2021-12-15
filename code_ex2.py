import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

MIN_RANGE = 0
MAX_RANGE = 1
DIM = 1


def get_uniform_data(num_of_points, range_x, range_y):
    data_x = np.random.uniform(range_x[MIN_RANGE], range_x[MAX_RANGE], (DIM, num_of_points))
    data_y = np.random.uniform(range_y[MIN_RANGE], range_y[MAX_RANGE], (DIM, num_of_points))
    return data_x, data_y


def get_gaussian_data(centers, std, num_of_points):
    data = list()
    for center in centers:
        data_x = np.random.normal(center, std * center, num_of_points)
        data_y = np.random.normal(-center, std * center, num_of_points)
        data.append((data_x, data_y))
    return data
    #     plt.scatter(data_x, data_y)
    # plt.show()


def get_moon_b(range_x_1, range_x_2):
    data_x_1 = np.random.uniform(range_x_1[0], range_x_1[1], 250)
    data_x_2 = np.random.uniform(range_x_2[0], range_x_2[1], 250)

    data_y_1 = 1 - data_x_1 ** 2
    data_y_1 = np.sqrt(data_y_1) + np.random.uniform(-0.1, 0.1, 250)
    data_y_2 = 1 - (data_x_2 - 1) ** 2
    data_y_2 = -1 * np.sqrt(data_y_2) + 0.5 + np.random.uniform(-0.1, 0.1, 250)

    data_x_1 += np.random.uniform(-0.1, 0.1, 250)
    data_x_2 += np.random.uniform(-0.1, 0.1, 250)

    plt.scatter(data_x_1, data_y_1)
    plt.scatter(data_x_2, data_y_2)
    plt.ylim((-1.5, 2.2))
    plt.xlim((-1.5,2.2))
    plt.show()


def get_moon_c(range_x_1, range_x_2):
    data_x_1 = np.random.uniform(range_x_1[0], range_x_1[1], 250)
    data_x_2 = np.random.uniform(range_x_2[0], range_x_2[1], 250)

    data_y_1 = 1 - data_x_1 ** 2
    data_y_1 = np.sqrt(data_y_1) + np.random.uniform(-0.4, -0.1, 250)
    data_y_2 = 1 - (data_x_2 - 1) ** 2
    data_y_2 = -1 * np.sqrt(data_y_2) + 0.5 + np.random.uniform(0.1, 0.4, 250)

    plt.scatter(data_x_1, data_y_1)
    plt.scatter(data_x_2, data_y_2)
    plt.ylim((-1.5, 2.5))
    plt.xlim((-1.5,2.5))
    plt.show()


def get_horizontal_clamps(centers, std_x, std_y, num_of_points):
    # (0,0) , (5,0), (0,2), (5,2)
    data = list()
    for center in centers:
        data_x = np.random.normal(center[0], std_x, num_of_points)
        data_y = np.random.normal(center[1], std_y, num_of_points)
        data.append((data_x, data_y))
        plt.scatter(data_x, data_y)
        plt.xlim((-3, 9))
        plt.ylim((-3, 7))
    plt.show()


def plot_data_as_scatter(x_array, y_array):
    plt.scatter(x_array, y_array)
    plt.show()


if __name__ == '__main__':
    # x, y = get_uniform_data(5)
    # plot_data(x, y)
    # get_gaussian_data([1, 2, 4], 0.5, 500)
    # get_horizontal_clamps([(0,0) , (5,0), (0,2), (5,2)], 1, 0.25, 125)
    get_moon_b([-1, 1], [0, 2])
    # x = np.linspace(-1.0, 1.0, 100)
    # y = np.linspace(-1.0, 1.0, 100)
    # X, Y = np.meshgrid(x,y)
    # F = X**2 + Y**2 - 1
    # plt.contour(X,Y,F,[0])
    # plt.show()