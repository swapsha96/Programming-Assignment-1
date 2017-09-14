import csv
import math
import matplotlib.pyplot as plt
from utils import *


def calculate_sigma_matrix(cov):
    avg_sigma = [[0 for x in range(len(cov[0][0]))] for y in range(len(cov[0]))]
    for matrix in cov:
        avg_sigma = add_matrices(avg_sigma, matrix)
    return divide_matrix_by_x(avg_sigma, len(cov))

def calculate_w(mi, sigma, pi):
    wi = multiply_matrices(inverse(sigma), mi)
    wi0 = math.log(pi) - divide_matrix_by_x(multiply_matrices(multiply_matrices(transpose(mi), inverse(sigma)), mi), 2)[0][0]
    return [wi, wi0]

def calculate_gix(wi, wi0, x):
    return multiply_matrices(transpose(wi), x)[0][0] + wi0

def gx(w1, w2, w10, w20, x):
    g1x = calculate_gix(w1, w10, x)
    g2x = calculate_gix(w2, w20, x)
    return (g1x - g2x)


if __name__ == '__main__':
    ratio = 0.75
    dirname = './sample/'
    names = [dirname + 'Class1.txt', dirname + 'Class2.txt']
    
    all_datasets, all_training_sets, all_test_sets, all_means, all_covariance_matrices, all_probabilities = generate_summary_from_classes(names, ratio, diagonal = False)

    sigma_matrix = calculate_sigma_matrix(all_covariance_matrices)
    param1 = calculate_w(transpose([all_means[0]]), sigma_matrix, all_probabilities[0])
    param2 = calculate_w(transpose([all_means[1]]), sigma_matrix, all_probabilities[1])

    x_min, x_max, y_min, y_max = get_graph_boundary(all_datasets[0] + all_datasets[1])

    plot_decision_boundary(w1, w2, w10, w20, x_min, x_max, y_min, y_max)

    plot_dataset(all_training_sets[0] + all_training_sets[1], 'g')
    plot_dataset(all_test_sets[0], 'k')
    plot_dataset(all_test_sets[1], 'w')


    plt.show()