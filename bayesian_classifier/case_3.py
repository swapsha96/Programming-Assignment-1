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
        cov[0][1] += (row[0] - means[0]) * (row[1] - means[1])
    cov[0][0] /= len(data)
    cov[1][1] /= len(data)
    cov[0][1] /= len(data)
    cov[1][0] = cov[0][1]
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
    for i in range(len(m)):
        for j in range(len(m[0])):
            m[i][j] = m[i][j]/x
    return m

def plot_dataset(dataset, color):
    x, y = [], []
    for row in dataset:
        x.append(row[0])
        y.append(row[1])
    plt.scatter(x, y, s= 2, marker ='o', c=color, alpha=1)

def calculate_slope(m1, m2):
    m = (m2[1] - m1[1])/(m2[0] - m1[0])
    return -1* (1/m)

def calculate_w(m1, m2, var, p1, p2):
    w1 = divide_matrix_by_x(transpose([m1]), var)
    w2 = divide_matrix_by_x(transpose([m2]), var)

    pdt1 = multiply_matrices([m1], transpose([m1]))
    pdt2 = multiply_matrices([m2], transpose([m2]))
    w10 = math.log(p1) - (pdt1[0][0]/(2*var))
    w20 = math.log(p2) - (pdt2[0][0]/(2*var))

    return w1, w2, w10, w20

def gx(w1, w2, w10, w20, x):
    g1x = multiply_matrices(transpose(w1), x)[0][0] + w10
    g2x = multiply_matrices(transpose(w2), x)[0][0] + w20
    return (g1x -g2x)

def get_graph_boundary(dataset1, dataset2):
    param1 = [[0, 0], [0, 0]]
    for i, col in zip(range(len(dataset1[0])), zip(*dataset1, *dataset2)):
        param1[i][0] = min(col)
        param1[i][1] = max(col)
    return param1[0][0], param1[0][1], param1[1][0], param1[1][1]

def plot_line(m, x0):
    x_0, y_0 = x0[0], x0[1]
    c = y_0 - m*x_0

    x = [x_0]
    y = [y_0]

    for i in range(20):
        x.append(i)
        y.append(m*i+c)

    plt.plot(x, y, 'k')

if __name__ == '__main__':
    ratio = 0.75
    names = ['Class1.txt', 'Class2.txt', 'Class3.txt']
    all_training_sets = []
    all_means = []
    sigma_sq = float(0)
    all_covariance_matrices = []
    total_data_size = 0
    all_probabilities = []


    for filename in names:
        dataset = load_dataset_from_file(filename)
        training_set, test_set = split_dataset(dataset, ratio)
        all_training_sets.append(training_set) # store t set
        means = mean(training_set)
        all_means.append(means)

        covariance_matrix = calculate_covariance_matrix(training_set, means)
        all_covariance_matrices.append(covariance_matrix)

        total_data_size += len(training_set)
        all_probabilities.append(len(training_set))

    for i in range(len(all_probabilities)):
        all_probabilities[i] /= total_data_size


    avg_covariance_matrix = divide_matrix_by_x(add_matrices(add_matrices(all_covariance_matrices[0], all_covariance_matrices[1]), all_covariance_matrices[2]), 3)
    sigma_sq = modulus(avg_covariance_matrix)
    w1, w2, w10, w20 = calculate_w(all_means[0], all_means[1], sigma_sq, all_probabilities[0], all_probabilities[1])

    x_min, x_max, y_min, y_max = get_graph_boundary(all_training_sets[0], all_training_sets[1])

    num_of_points = 500
    x_min -= 1
    x_max += 1
    y_min -= 1
    y_max += 1
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
    cls1_x = []
    cls1_y = []
    cls2_x = []
    cls2_y = []
    for x_i, y_i in zip(x, y):
        count += 1
        x_matrix = [[x_i], [y_i]]
        fx = gx(w1, w2, w10, w20, x_matrix)
        print(count, fx)
        if fx >= 0:
            cls1_x.append(x_i)
            cls1_y.append(y_i)
        else:
            cls2_x.append(x_i)
            cls2_y.append(y_i)

    plt.scatter(cls1_x, cls1_y, s= 3, marker ='o', c='r', alpha=1)
    plt.scatter(cls2_x, cls2_y, s= 3, marker ='o', c='b', alpha=1)
    plot_dataset(all_training_sets[0], 'g')
    plot_dataset(all_training_sets[1], 'g')

    # plot_dataset(all_training_sets[0], 'r')
    # plot_dataset(all_training_sets[1], 'b')

    # x0 = calculate_x0(all_means[1], all_means[2], sigma_sq, all_probabilities[1], all_probabilities[2])
    # plot_line(slope, x0)

    # x0 = calculate_x0(all_means[2], all_means[0], sigma_sq, all_probabilities[2], all_probabilities[0])
    # plot_line(slope, x0)

    plt.show()