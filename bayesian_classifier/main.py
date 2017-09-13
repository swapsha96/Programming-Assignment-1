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

def tranpose(matrix):
	return [[matrix[0]], [matrix[1]]]

def multiply_matrices(m1, m2):
	return [[m1[0][0] * m2[0], m1[0][0] * m2[1]], [m1[1][0] * m2[0], m1[1][0] * m2[1]]]

def add_matrices(m1, m2):
	return [[m1[0][0] + m2[0][0], m1[0][1] + m2[0][1]], [m1[0][0] + m2[0][0], m1[0][1] + m2[0][1]]]

def divide_matrix_by_x(m, x):
	return [[m[0][0]/x, m[0][1]/x], [m[1][0]/x, m[1][1]/x]]

def plot_dataset(dataset):
	x, y = [], []
	for row in dataset:
		x.append(row[0])
		y.append(row[1])
	plt.scatter(x, y, c='r', alpha=0.5)

def calculate_slope(m1, m2):
	m = (m2[1] - m1[1])/(m2[0] - m1[0])
	return -1* (1/m)

def calculate_x0(m1, m2, var, p1, p2):
	diff = [m1[0] - m2[0], m1[1]- m2[1]]
	diff_t = tranpose(diff)
	pdt = multiply_matrices(diff_t, diff)
	norm = modulus(pdt)
	x01 = (m1[0] + m2[0])/2 - (var*math.log(p1/p2)/norm)*(m1[0]-m2[0])
	x02 = (m1[1] + m2[1])/2 - (var*math.log(p1/p2)/norm)*(m1[1]-m2[1])
	return [x01, x02]

def plot_line(slope, x0):
	x, y = x0[0], x0[1]
	c = y - slope*x
	x2 = 1
	y2 = slope*x2 +c
	plt.plot([x, x2], [y, y2], 'k')

if __name__ == '__main__':
	ratio = 0.75
	names = ['Class1.txt', 'Class2.txt', 'Class3.txt']
	all_training_sets = []
	all_means = []
	sigma_sq = float(0)
	all_covariance_matrices = []
	all_probabilities = []

	for filename in names:
		dataset = load_dataset_from_file(filename)
		training_set, test_set = split_dataset(dataset, ratio)
		all_training_sets.append(training_set) # store t set
		plot_dataset(training_set)
		means = mean(training_set)
		all_means.append(means)

		covariance_matrix = calculate_covariance_matrix(training_set, means)
		all_covariance_matrices.append(covariance_matrix)

		probability_wi = 1/3
		all_probabilities.append(probability_wi)

	cov_matrix = add_matrices(all_covariance_matrices[0], all_covariance_matrices[1])
	cov_matrix = add_matrices(cov_matrix, all_covariance_matrices[2])
	cov_matrix = divide_matrix_by_x(covariance_matrix, 3)
	sigma_sq = modulus(cov_matrix)

	slope = calculate_slope(all_means[0], all_means[1])
	x0 = calculate_x0(all_means[0], all_means[1], sigma_sq, all_probabilities[0], all_probabilities[1])
	plot_line(slope, x0)

	slope = calculate_slope(all_means[1], all_means[2])
	x0 = calculate_x0(all_means[1], all_means[2], sigma_sq, all_probabilities[1], all_probabilities[2])
	plot_line(slope, x0)

	slope = calculate_slope(all_means[2], all_means[0])
	x0 = calculate_x0(all_means[2], all_means[0], sigma_sq, all_probabilities[2], all_probabilities[0])
	plot_line(slope, x0)

	plt.show()