import csv
import math
import matplotlib.pyplot as plt
from utils import *


def calculate_sigma_sq(cov):
    avg_sigma = []
    for matrix in cov:
        avg_sigma.append((matrix[0][0] + matrix[1][1]) / 2)
    return (sum(avg_sigma) / len(avg_sigma))

def calculate_w(mi, var, pi):
    wi = divide_matrix_by_x(mi, var)
    wi0 = math.log(pi) - (multiply_matrices(transpose(mi), mi)[0][0] / (2 * var))
    return [wi, wi0]

def calculate_gix(param, x):
    wi, wi0 = tuple(param)
    return multiply_matrices(transpose(wi), x)[0][0] + wi0

def gx_argmax(params, x):
    gix = []
    for param in params:
        gix.append(calculate_gix(param, x))
    return gix.index(max(gix))


if __name__ == '__main__':
    ratio = 0.75
    dirname = './LS_Group16/'
    names = [dirname + 'Class1.txt', dirname + 'Class2.txt', dirname + 'Class3.txt']
    
    all_datasets, all_training_sets, all_test_sets, all_means, all_covariance_matrices, all_probabilities = generate_summary_from_classes(names, ratio, diagonal = True)

    sigma_sq = calculate_sigma_sq(all_covariance_matrices)
    params = []
    for i in range(len(all_datasets)):
        params.append(calculate_w(transpose([all_means[i]]), sigma_sq, all_probabilities[i]))

    complete_dataset = []
    for dataset in all_datasets:
        complete_dataset += dataset
    margin = get_graph_boundary(complete_dataset)

    plot_decision_boundary(params, margin, gx_argmax)
    plot_dataset(all_training_sets[0] + all_training_sets[1] + all_training_sets[2], 'm')
    # plot_dataset(all_test_sets[0], 'k')
    # plot_dataset(all_test_sets[1], 'w')

    plt.show()