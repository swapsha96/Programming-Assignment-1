import csv
import math
import matplotlib.pyplot as plt
import numpy as np


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

def calculate_covariance_matrix(data, means):
    cov = [[0, 0], [0, 0]]
    for row in data:
        cov[0][0] += ((row[0] - means[0]) ** 2)
        cov[1][1] += ((row[1] - means[1]) ** 2)
        # cov[0][1] += (row[0] - means[0]) * (row[1] - means[1])
    cov[0][0] /= len(data)
    cov[1][1] /= len(data)
    # cov[0][1] /= len(data)
    # cov[1][0] = cov[0][1]
    return cov

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

def plot_dataset(dataset, color):
    x, y = [], []
    for row in dataset:
        x.append(row[0])
        y.append(row[1])
    plt.scatter(x, y, s= 2, marker ='o', c=color, alpha=1)

def calculate_sigma_sq(cov):
    avg_sigma = []
    for matrix in cov:
        avg_sigma.append((matrix[0][0] + matrix[1][1]) / 2)

    return (sum(avg_sigma) / len(avg_sigma))

def calculate_w(mi, var, pi):
    wi = divide_matrix_by_x(mi, var)
    wi0 = math.log(pi) - (multiply_matrices(transpose(mi), mi)[0][0] / (2 * var))

    return wi, wi0

def generate_summary_from_classes(filenames):
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
        all_means.append(means)

        covariance_matrix = calculate_covariance_matrix(training_set, means)
        all_covariance_matrices.append(covariance_matrix)

        total_data_size += len(training_set)
        all_probabilities.append(len(training_set))

    for i in range(len(all_probabilities)):
        all_probabilities[i] /= total_data_size

    return all_datasets, all_training_sets, all_test_sets, all_means, all_covariance_matrices, all_probabilities

def calculate_gix(wi, wi0, x):
    return multiply_matrices(transpose(wi), x)[0][0] + wi0

def gx(w1, w2, w10, w20, x):
    g1x = calculate_gix(w1, w10, x)
    g2x = calculate_gix(w2, w20, x)
    return (g1x - g2x)
    
def get_graph_boundary(dataset):
    param = [[0, 0], [0, 0]]
    for i, col in zip(range(len(dataset[0])), zip(*dataset)):
        param[i][0], param[i][1] = min(col), max(col)

    x_min, x_max, y_min, y_max = param[0][0], param[0][1], param[1][0], param[1][1]
    x_min -= 1
    x_max += 1
    y_min -= 1
    y_max += 1
    return x_min, x_max, y_min, y_max

def plot_decision_boundary(w1, w2, w10, w20, x_min, x_max, y_min, y_max):
    num_of_points = 500
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

    count = 0
    cls1_x, cls1_y, cls2_x, cls2_y = [], [], [], []

    for x_i, y_i in zip(x, y):
        count += 1
        x_matrix = [[x_i], [y_i]]
        fx = gx(w1, w2, w10, w20, x_matrix)
        if fx >= 0:
            cls1_x.append(x_i)
            cls1_y.append(y_i)
        else:
            cls2_x.append(x_i)
            cls2_y.append(y_i)

    plt.scatter(cls1_x, cls1_y, s= 3, marker ='o', c='r', alpha=1)
    plt.scatter(cls2_x, cls2_y, s= 3, marker ='o', c='b', alpha=1)

if __name__ == '__main__':
    ratio = 0.75
    dirname = './sample/'
    names = [dirname + 'Class1.txt', dirname + 'Class2.txt']
    
    all_datasets, all_training_sets, all_test_sets, all_means, all_covariance_matrices, all_probabilities = generate_summary_from_classes(names)

    sigma_sq = calculate_sigma_sq(all_covariance_matrices)
    w1, w10= calculate_w(transpose([all_means[0]]), sigma_sq, all_probabilities[0])
    w2, w20= calculate_w(transpose([all_means[1]]), sigma_sq, all_probabilities[1])

    x_min, x_max, y_min, y_max = get_graph_boundary(all_datasets[0] + all_datasets[1])

    plot_decision_boundary(w1, w2, w10, w20, x_min, x_max, y_min, y_max)

    plot_dataset(all_training_sets[0] + all_training_sets[1], 'g')
    plot_dataset(all_test_sets[0], 'k')
    plot_dataset(all_test_sets[1], 'w')


    plt.show()