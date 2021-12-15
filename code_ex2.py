import itertools

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


MIN_RANGE = 0
MAX_RANGE = 1
DIM = 1


def get_uniform_data(num_of_points, plot):
    data_x = np.random.uniform(-10, 2, (DIM, num_of_points))
    data_y = np.random.uniform(18, 45, (DIM, num_of_points))
    if plot:
        plt.scatter(data_x, data_y)
        plt.show()
    data = merge_x_y(data_x.reshape(1, 500).tolist()[0], data_y.reshape(1, 500).tolist()[0])
    return data


def get_gaussian_data(centers, std, num_of_points, plot):
    data = list()
    for center in centers:
        data_x = np.random.normal(center, std * center, num_of_points)
        data_y = np.random.normal(-center, std * center, num_of_points)
        temp_data = merge_x_y(data_x.tolist(), data_y.tolist())
        if plot:
            plt.scatter(data_x, data_y)
            plt.show()
        data.append(temp_data)
    data = np.array(data)
    return data


def get_moon_b(range_x_1, range_x_2, plot):
    data_x_1 = np.random.uniform(range_x_1[0], range_x_1[1], 250)
    data_x_2 = np.random.uniform(range_x_2[0], range_x_2[1], 250)

    data_y_1 = 1 - data_x_1 ** 2
    data_y_1 = np.sqrt(data_y_1) + np.random.uniform(-0.1, 0.1, 250)
    data_y_2 = 1 - (data_x_2 - 1) ** 2
    data_y_2 = -1 * np.sqrt(data_y_2) + 0.5 + np.random.uniform(-0.1, 0.1, 250)

    data_x_1 += np.random.uniform(-0.1, 0.1, 250)
    data_x_2 += np.random.uniform(-0.1, 0.1, 250)
    if plot:
        plt.scatter(data_x_1, data_y_1)
        plt.scatter(data_x_2, data_y_2)
        plt.ylim((-1.5, 2.2))
        plt.xlim((-1.5, 2.2))
        plt.show()
    else:
        data_y = np.concatenate([data_y_1, data_y_2])
        data_x = np.concatenate([data_x_1, data_x_2])
        data = merge_x_y(data_x.tolist(), data_y.tolist())
        return data


def get_moon_c(range_x_1, range_x_2, plot):
    data_x_1 = np.random.uniform(range_x_1[0], range_x_1[1], 250)
    data_x_2 = np.random.uniform(range_x_2[0], range_x_2[1], 250)

    data_y_1 = 1 - data_x_1 ** 2
    data_y_1 = np.sqrt(data_y_1) + np.random.uniform(-0.4, -0.1, 250)
    data_y_2 = 1 - (data_x_2 - 1) ** 2
    data_y_2 = -1 * np.sqrt(data_y_2) + 0.5 + np.random.uniform(0.1, 0.4, 250)

    if plot:
        plt.scatter(data_x_1, data_y_1)
        plt.scatter(data_x_2, data_y_2)
        plt.ylim((-1.5, 2.2))
        plt.xlim((-1.5, 2.2))
        plt.show()
    else:
        data_y = np.concatenate([data_y_1, data_y_2])
        data_x = np.concatenate([data_x_1, data_x_2])
        data = merge_x_y(data_x.tolist(), data_y.tolist())
        return data


def get_letters_data(plot):
    # c plot:
    data_x_1 = np.random.uniform(-2, 0, 250)
    data_y_1 = 1 - (data_x_1 + 1) ** 2
    data_y_1 = np.concatenate([np.sqrt(data_y_1), -np.sqrt(data_y_1)])
    data_x_1 = np.concatenate([data_x_1, data_x_1])
    data_y_1 = data_y_1[data_x_1 < -0.5]
    data_y_1 += np.random.uniform(-0.1, 0.1, data_y_1.size)
    data_x_1 = data_x_1[data_x_1 < - 0.5]
    data_x_1 += np.random.uniform(-0.1, 0.1, data_x_1.size)
    plt.scatter(data_x_1, data_y_1)
    # s plot firs c:
    data_x_2 = np.random.uniform(0, 2, 125)
    data_y_2 = 1 - (data_x_2 - 1) ** 2
    data_y_2 = np.concatenate([np.sqrt(data_y_2), -np.sqrt(data_y_2)]) * 0.5 + 0.5
    data_x_2 = np.concatenate([data_x_2, data_x_2])
    data_y_2 = data_y_2[data_x_2 < 1.5]
    data_y_2 += np.random.uniform(-0.1, 0.1, data_y_2.size)
    data_x_2 = data_x_2[data_x_2 < 1.5]
    data_x_2 += np.random.uniform(-0.1, 0.1, data_x_2.size)
    # s plot second c :
    data_x_3 = np.random.uniform(0, 2, 125)
    data_y_3 = 1 - (data_x_3 - 1) ** 2
    data_y_3 = np.concatenate([np.sqrt(data_y_3), -np.sqrt(data_y_3)]) * 0.5 - 0.5
    data_x_3 = np.concatenate([data_x_3, data_x_3])
    data_y_3 = data_y_3[data_x_3 < 1.5]
    data_y_3 += np.random.uniform(-0.1, 0.1, data_y_3.size)
    data_x_3 = data_x_3[data_x_3 < 1.5]
    data_x_3 += np.random.uniform(-0.1, 0.1, data_x_3.size)
    data_x_3 = data_x_3 * -1 + 1.9
    data_x_2 = np.concatenate([data_x_2, data_x_3])
    data_y_2 = np.concatenate([data_y_2, data_y_3])
    # plot
    if plot:
        plt.scatter(data_x_2, data_y_2)
        plt.ylim((-3, 3))
        plt.xlim((-3, 3))
        plt.show()
    else:
        data_y = np.concatenate([data_y_1, data_y_2])
        data_x = np.concatenate([data_x_1, data_x_2])
        data = merge_x_y(data_x.tolist(), data_y.tolist())
        return data


def get_horizontal_clamps(centers, std_x, std_y, num_of_points, plot):
    # (0,0) , (5,0), (0,2), (5,2)
    data_x = np.array([])
    data_y = np.array([])
    for center in centers:
        data_x = np.concatenate([data_x, np.random.normal(center[0], std_x, num_of_points)])
        data_y = np.concatenate([data_y, np.random.normal(center[1], std_y, num_of_points)])
    data = merge_x_y(data_x.tolist(), data_y.tolist())
    if plot:
        plt.scatter(data_x, data_y)
        plt.xlim((-3, 9))
        plt.ylim((-3, 7))
        plt.show()
    else:
        return data


def run_KMean(data, k, random_state, type_of_data, algorithm='auto'):
    kmeans = KMeans(n_clusters=k, random_state=random_state, algorithm=algorithm).fit_predict(data)
    plt.title(f'{type_of_data} clustering, k = {k} ,algorithm = {algorithm}')
    plt.scatter(data[:, 0], data[:, 1], c=kmeans)
    plt.savefig(f'figures/{type_of_data}_cluster_k-{k}',bbox_inches='tight')
    plt.show()
    return f'figures/{type_of_data}_cluster_k-{k}'


def merge_x_y(x_data, y_data):
    data = list()
    for i in range(len(x_data)):
        data.append([x_data[i], y_data[i]])
    return np.array(data)




if __name__ == '__main__':
    # run_KMean(get_uniform_data(500, plot=False), 4, 170)
    # run_KMean(get_gaussian_data([1, 2, 4], 0.5, 500, plot=False)[2], 4, 170)
    # run_KMean(get_horizontal_clamps([(0,0) , (5,0), (0,2), (5,2)], 1, 0.25, 125, False), 5, 170)
    # run_KMean(get_moon_b([-1, 1], [0, 2], False),  2, 170)
    # run_KMean(get_moon_c([-1, 1], [0, 2], False),  2, 170)
    run_KMean(get_letters_data(False), 3, 170, 'letters distribution')
    # x, y = get_uniform_data(500, True)
    # plot_data(x, y)
    # get_gaussian_data([1, 2, 4], 0.5, 500)
    # get_horizontal_clamps([(0,0) , (5,0), (0,2), (5,2)], 1, 0.25, 125)
    # get_moon_b([-1, 1], [0, 2])
    # get_letters_data()
