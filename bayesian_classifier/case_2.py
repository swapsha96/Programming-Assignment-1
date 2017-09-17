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

    print("Case 2: " + (" vs ".join(class_names)))
    
    all_datasets, all_training_sets, all_test_sets, all_means, all_covariance_matrices, all_probabilities = generate_summary_from_classes(class_addresses, ratio, diagonal = True)

    sigma_matrix = calculate_sigma_matrix(all_covariance_matrices)
    params = []
    for i in range(len(all_datasets)):
        params.append(calculate_w(all_means[i], sigma_matrix, all_probabilities[i]))

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
    
    print("Generating and plotting graphs.")
    plt.figure(1)
    plot_decision_boundary(calculate_gix, params, margin)
    plot_dataset(class_names, all_training_sets)
    plt.legend(loc='upper right')
    plt.savefig(dirname + "Case2 " + (" vs ".join(class_names)) + ".png", format='png')
    
    plt.figure(2)
    plot_contour_graph(calculate_gix, params, margin)
    plot_dataset(class_names, all_training_sets)
    plt.legend(loc='upper right')
    plt.savefig(dirname + "Case2 " + (" vs ".join(class_names)) + " contour.png", format='png')
    print("Done.")
