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


if __name__ == '__main__':
    ratio = 0.75
    dirname = ""
    # dirname = './LS_Group16/'
    # dirname = './NLS_Group16/'
    # dirname = './RD_Group16/'
    class_names = []
    class_addresses = []
    if len(sys.argv) < 4:
        print("You need to enter at least 4 parameters.\n(Format: ./[case_x].py ./dirname/ Class1.txt Class2.txt")
        exit()
    else:
        dirname = sys.argv[1]
        for i in range(2, len(sys.argv)):
            class_names.append(sys.argv[i])
            class_addresses.append(dirname + sys.argv[i])
    
    all_datasets, all_training_sets, all_test_sets, all_means, all_covariance_matrices, all_probabilities = generate_summary_from_classes(class_addresses, ratio, diagonal = True)

    sigma_sq = calculate_sigma_sq(all_covariance_matrices)
    params = []
    for i in range(len(all_datasets)):
        params.append(calculate_w(all_means[i], sigma_sq, all_probabilities[i]))

    complete_dataset = []
    for dataset in all_datasets:
        complete_dataset += dataset
    margin = get_graph_boundary(complete_dataset)

    confusion_matrix = generate_confusion_matrix(calculate_gix, params, all_test_sets)
    print("Confusion Matrix: ")
    print_matrix(confusion_matrix)

    print("Classification Accuracy: " + str(get_accuracy(confusion_matrix) * 100) + "%")

    print("Class Precisions:")
    precision = get_precision(confusion_matrix)
    for name, p in zip(class_names, precision):
        print("\t" + name + ": " + str(p * 100) + "%")
    print("Mean Precision: " + str(get_mean_precision(confusion_matrix) * 100) + "%")

    print("Class Recalls:")
    recall = get_recall(confusion_matrix)
    for name, r in zip(class_names, recall):
        print("\t" + name + ": " + str(r * 100) + "%")
    print("Mean Recall: " + str(get_mean_recall(confusion_matrix) * 100) + "%")

    print("Class F-measures:")
    f_measure = get_f_measure(get_precision(confusion_matrix), get_recall(confusion_matrix))
    for name, f in zip(class_names, f_measure):
        print("\t" + name + ": " + str(f * 100) + "%")
    print("Mean F-measure: " + str(get_mean_f_measure(get_precision(confusion_matrix), get_recall(confusion_matrix)) * 100) + "%")
    
    print("Building and saving graph.")
    plt.figure(1)
    plot_decision_boundary(calculate_gix, params, margin)
    plot_dataset(class_names, all_training_sets)
    plt.legend(loc='upper right')
    plt.savefig(dirname + "Case1 " + (" vs ".join(class_names)) + ".png", format='png')
    
    plt.figure(2)
    plot_contour_graph(calculate_gix, params, margin)
    plot_dataset(class_names, all_training_sets)
    plt.legend(loc='upper right')
    plt.savefig(dirname + "Case1 " + (" vs ".join(class_names)) + " contour.png", format='png')
    print("Done.")
