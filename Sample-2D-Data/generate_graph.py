import csv
import math
import matplotlib.pyplot as plt
import numpy as np


def load_dataset_from_file(filename):
	return list(csv.reader(open(filename, 'r'), delimiter = ' '))

def split_dataset(dataset, ratio):
	r = int(ratio * len(dataset))
	return (dataset[:r, :], dataset[r:, :])


def summarize_dataset(dataset):
	summary = np.column_stack((np.mean(dataset, axis = 0), np.std(dataset, axis = 0)))
	return summary

def calculate_probabilty(x, mean, std):
	return (1 / (math.sqrt(2 * math.pi) * std)) * (math.exp(-(math.pow(x - mean, 2) / (2 * math.pow(std, 2)))))

def calculate_class_probability(dataset, summary):
	probability = 1
	for column, col_summary in zip(dataset, summary):
		for x in column:
			probability *= calculate_probabilty(x, col_summary[0], col_summary[1])
		return probability


if __name__ == '__main__':
	ratio = 0.75
	dataset = load_dataset_from_file('Class1.txt')
	training_set, test_set = split_dataset(dataset, ratio)
	summary = summarize_dataset(training_set)
	probability = calculate_class_probability(test_set, summary)
	print(probability)
