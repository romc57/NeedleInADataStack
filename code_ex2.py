import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering

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
    data_x = np.array(list())
    data_y = np.array(list())
    first = True
    for center in centers:
        data_x = np.concatenate([data_x, np.random.normal(center, std * center, num_of_points)])
        data_y = np.concatenate([data_y, np.random.normal(-center, std * center, num_of_points)])
        temp_data = merge_x_y(data_x.tolist(), data_y.tolist())
        if first:
            data = temp_data[:]
            first = False
        else:
            data = np.concatenate([data, temp_data])
    if plot:
        plt.scatter(data_x, data_y)
        plt.show()
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
    data_y_1 = np.sqrt(data_y_1) + np.random.uniform(-0.4, -0.3, 250)
    data_y_2 = 1 - (data_x_2 - 1) ** 2
    data_y_2 = -1 * np.sqrt(data_y_2) + 0.5 + np.random.uniform(0.3, 0.4, 250)

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


def run_KMean(data, k, random_state, type_of_data, show=True, save=False, algorithm='auto'):
    kmeans = KMeans(n_clusters=k, random_state=random_state, algorithm=algorithm).fit_predict(data)
    plt.title(f'{type_of_data} clustering, k = {k} ,algorithm = {algorithm}')
    plt.scatter(data[:, 0], data[:, 1], c=kmeans)
    if save:
        plt.savefig(f'clusters/{type_of_data}_cluster_k-{k}', bbox_inches='tight')
    if show:
        plt.show()
    return f'clusters/{type_of_data}_cluster_k-{k}.png'


def plot_synthetic_data(data, type_of_data, filename, save=False, show=True):
    plt.title(f'{type_of_data} - raw data')
    plt.scatter(data[:, 0], data[:, 1], c=['#000000'])
    if save:
        plt.savefig(f'figures/{filename}.png', bbox_inches='tight')
    if show:
        plt.show()
    plt.clf()
    return f'figures/{filename}.png'


def run_problem_3():
    output_dic = {'uniform_data': list(), 'gaussian_data': list(), 'clumps': list(), 'names_letters': list(),
                  'moon_b': list(), 'moon_c': list()}
    for i in range(2):
        uniform_data = get_uniform_data(500, plot=False)
        output_dic['uniform_data'].append(
            {'path': plot_synthetic_data(uniform_data, f'Uniform data {i}', f'uniform{i}', save=True,
                                         show=False), 'data': uniform_data})
        centers = [1, 2, 4]
        gaussian_data = get_gaussian_data(centers, 0.5, 500, plot=False)
        output_dic['gaussian_data'].append(
            {'path': plot_synthetic_data(gaussian_data, f'gaussian data {i}, centers : {centers}',
                                         f'gaussian{i}', save=True, show=False), 'data': gaussian_data[i]})
        names_data = get_letters_data(plot=False)
        output_dic['names_letters'].append(
            {'path': plot_synthetic_data(names_data, f'Last Names first letters data {i}', f'letters{i}',
                                         save=True, show=False), 'data': names_data})
        clumps_data = get_horizontal_clamps([(0, 0), (5, 0), (0, 2), (5, 2)], 1, 0.25, 125, plot=False)
        output_dic['clumps'].append(
            {'path': plot_synthetic_data(clumps_data, f'Four horizontal clumps data {i}', f'clumps{i}',
                                         save=True, show=False), 'data': clumps_data})
        moon_b = get_moon_b([-1, 1], [0, 2], plot=False)
        output_dic['moon_b'].append(
            {'path': plot_synthetic_data(moon_b, f'Unconnected Moons  {i}', f'moon_b_{i}',
                                         save=True, show=False), 'data': moon_b})
        moon_c = get_moon_c([-1, 1], [0, 2], plot=False)
        output_dic['moon_c'].append(
            {'path': plot_synthetic_data(moon_c, f'Connected Moons  {i}', f'moon_c_{i}',
                                         save=True, show=False), 'data': moon_c})
    return output_dic


def hierarchical_clustering(data, type_of_data, save=False, show=True):
    figures = list()
    for linkage in ("average", "complete", "single"):
        for k in [2, 3, 4, 5]:
            clustering = AgglomerativeClustering(linkage=linkage, n_clusters=k).fit_predict(data)
            plt.title(f'{type_of_data} hierarchical clustering, k = {k} ,algorithm = {linkage}')
            plt.scatter(data[:, 0], data[:, 1], c=clustering)
            if save:
                plt.savefig(f'clusters/{type_of_data}_cluster_k-{k}', bbox_inches='tight')
            if show:
                plt.show()
            figures.append(f'clusters/{type_of_data}_cluster_k-{k}.png')
    return figures


def merge_x_y(x_data, y_data):
    data = list()
    for i in range(len(x_data)):
        data.append([x_data[i], y_data[i]])
    return np.array(data)


if __name__ == '__main__':
    # run_KMean(get_uniform_data(500, plot=False), 4, 170)
    # run_KMean(get_gaussian_data([1, 2, 4], 0.5, 500, plot=False)[2], 4, 170)
    # run_KMean(get_horizontal_clamps([(0,0) , (5,0), (0,2), (5,2)], 1, 0.25, 125, False), 5, 170)
    # run_KMean(get_moon_b([-1, 1], [0, 2], False),  2, 170, 'moon', algorithm='elkan')
    # run_KMean(get_moon_c([-1, 1], [0, 2], False),  2, 170)

    # run_KMean(get_letters_data(False), 3, 170, 'letters distribution')
    # x, y = get_uniform_data(500, True)
    # plot_data(x, y)
    # get_gaussian_data([1, 2, 4], 0.5, 500)
    # get_horizontal_clamps([(0,0) , (5,0), (0,2), (5,2)], 1, 0.25, 125)
    # get_moon_b([-1, 1], [0, 2])
    print(run_problem_3())
    # hierarchical_clustering(get_moon_b([-1, 1], [0, 2], False), 'moon')
