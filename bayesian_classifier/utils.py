import csv
import math
import matplotlib.pyplot as plt


def load_dataset_from_file(filename):
    dataset = list(csv.reader(open(filename, 'r'), delimiter = ' '))
    dataset = [[float(x) for x in row] for row in dataset]
    return dataset

def split_dataset(dataset, ratio):
    r = int(ratio * len(dataset))
    return (dataset[:r], dataset[r:])

def mean(data):
    sum_x, sum_y = float(0), float(0)
    for row in data:
        sum_x += row[0]
        sum_y += row[1]
    return [float(sum_x/len(data)), float(sum_y/len(data))]

def modulus(matrix_2d):
    return (matrix_2d[0][0]*matrix_2d[1][1])-(matrix_2d[0][1]*matrix_2d[1][0])

def inverse(matrix):
    return [[matrix[1][1], -1*matrix[0][1]], [-1*matrix[1][0], matrix[0][0]]]

def transpose(m):
    return [[m[j][i] for j in range(len(m))] for i in range(len(m[0]))]

def multiply_matrices(m1, m2):
    m = [[] for _ in range(len(m1))]
    for i in range(len(m1)):        
        for j in range(len(m2[0])):
            sum = 0
            for k in range(len(m1[0])):
                sum += m1[i][k] * m2[k][j]
            m[i].append(sum)
    return m

def add_matrices(m1, m2):
    return [[m1[0][0] + m2[0][0], m1[0][1] + m2[0][1]], [m1[1][0] + m2[1][0], m1[1][1] + m2[1][1]]]

def divide_matrix_by_x(m, x):
    matrix = [[0 for x in range(len(m[0]))] for y in range(len(m))]
    for i in range(len(m)):
        for j in range(len(m[0])):
            matrix[i][j] = m[i][j]/x
    return matrix

def plot_dataset(datasets):
    colors = ['c', 'm', 'k', 'w']
    for i, dataset in zip(range(len(datasets)), datasets):
            plt.scatter(transpose(dataset)[0], transpose(dataset)[1], s = 2, marker ='o', c=colors[i], alpha=1)

def generate_summary_from_classes(filenames, ratio, diagonal = False):
    all_datasets, all_training_sets, all_test_sets, all_means, all_covariance_matrices, all_probabilities = [], [], [], [], [], []
    sigma_sq = float(0)
    total_data_size = int(0)

    for filename in filenames:
        dataset = load_dataset_from_file(filename)
        all_datasets.append(dataset)

        training_set, test_set = split_dataset(dataset, ratio)
        all_training_sets.append(training_set)
        all_test_sets.append(test_set)

        means = mean(training_set)
        all_means.append(transpose([means]))

        covariance_matrix = calculate_covariance_matrix(training_set, means, diagonal)
        all_covariance_matrices.append(covariance_matrix)

        total_data_size += len(training_set)
        all_probabilities.append(len(training_set))

    for i in range(len(all_probabilities)):
        all_probabilities[i] /= total_data_size

    return all_datasets, all_training_sets, all_test_sets, all_means, all_covariance_matrices, all_probabilities

def get_graph_boundary(dataset):
    param = [[0, 0], [0, 0]]
    for i, col in zip(range(len(dataset[0])), zip(*dataset)):
        param[i][0], param[i][1] = min(col), max(col)

    x_min, x_max, y_min, y_max = param[0][0], param[0][1], param[1][0], param[1][1]
    x_min -= 5
    x_max += 5
    y_min -= 5
    y_max += 5
    return [x_min, x_max, y_min, y_max]

def gx_argmax(calculate_gix, params, x):
    gix = []
    for param in params:
        gix.append(calculate_gix(param, x))
    return gix.index(max(gix))

def plot_decision_boundary(calculate_gix, params, margin):
    num_of_points = 500
    x_min, x_max, y_min, y_max = tuple(margin)
    x_diff = (x_max - x_min) / num_of_points
    y_diff = (y_max - y_min) / num_of_points

    x = []
    y = []

    for i in range(num_of_points):
        x_i = x_min + i * x_diff
        for j in range(num_of_points):
            y_i = y_min + j * y_diff
            x.append(x_i)
            y.append(y_i)

    cls_x, cls_y = [], []
    for i in range(len(params)):
        cls_x.append([])
        cls_y.append([])

    for x_i, y_i in zip(x, y):
        gx = gx_argmax(calculate_gix, params, [[x_i], [y_i]])
        cls_x[gx].append(x_i)
        cls_y[gx].append(y_i)

    colors = ['r', 'b', 'g', 'y']
    for i in range(len(params)):
        plt.scatter(cls_x[i], cls_y[i], s= 3, marker ='o', c=colors[i], alpha=1)

def calculate_covariance_matrix(data, means, diagonal = False):
    cov = [[0, 0], [0, 0]]
    for row in data:
        cov[0][0] += ((row[0] - means[0]) ** 2)
        cov[1][1] += ((row[1] - means[1]) ** 2)
        if diagonal == False:
            cov[0][1] += (row[0] - means[0]) * (row[1] - means[1])
    cov[0][0] /= len(data)
    cov[1][1] /= len(data)
    if diagonal == False:
        cov[0][1] /= len(data)
        cov[1][0] = cov[0][1]
    return cov

def generate_confusion_matrix(calculate_gix, params, datasets):
    matrix = [[0 for x in range(len(datasets))] for y in range(len(datasets))]
    for i, dataset in zip(range(len(datasets)), datasets):
        for x, y in zip(transpose(dataset)[0], transpose(dataset)[1]):
            gx = gx_argmax(calculate_gix, params, [[x], [y]])
            matrix[i][gx] += 1
    return matrix

def get_accuracy(confusion_matrix):
    num, denom = 0, 0
    for diagonal_index in range(len(confusion_matrix)):
        num += confusion_matrix[diagonal_index][diagonal_index]

    for i in range(len(confusion_matrix)):
        for j in range(len(confusion_matrix)):
            denom += confusion_matrix[i][j]

    return (num / denom)

def get_precision(confusion_matrix):
    precision = []
    for i, row in zip(range(len(confusion_matrix)), confusion_matrix):
        precision.append(confusion_matrix[i][i] / sum(row))
    return precision

def get_mean_precision(confusion_matrix):
    precision = get_precision(confusion_matrix)
    return (sum(precision) / len(precision))

def get_recall(confusion_matrix):
    recall = []
    for i, row in zip(range(len(confusion_matrix)), transpose(confusion_matrix)):
        recall.append(confusion_matrix[i][i] / sum(row))
    return recall

def get_mean_recall(confusion_matrix):
    recall = get_recall(confusion_matrix)
    return (sum(recall) / len(recall))

def get_f_measure(precision, recall):
    measure = []
    for p, r in zip(precision, recall):
        measure.append((2 * p * r) / (p + r))
    return measure

def get_mean_f_measure(precision, recall):
    f_measure = get_f_measure(precision, recall)
    return (sum(f_measure) / len(f_measure))
