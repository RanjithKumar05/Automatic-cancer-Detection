import csv
import math
import operator
import statistics
from statistics import multimode
import random


def load_from_csv(file_path: str) -> list:
    """load data from csv file and return matrix of list of lists.
    """
    data = []
    with open(file_path, newline="") as csvfile:
        reader = csv.reader(csvfile, delimiter=",")
        for row in reader:
            row = [int(i) for i in row]
            data.append(row)
    return data


def get_distance(list1: list, list2: list) -> float:
    """calculate euclidean distance between two lists
    """
    if len(list1) != len(list2):
        return """Length of both list should be same."""
    else:
        distance = math.sqrt(sum(
            [(value1 - value2) ** 2 for value1, value2 in zip(list1, list2)]
            ))
        return distance


def get_standard_deviation(data: list, col_num: int) -> float:
    """calculate standard deviation of column number passed as parameter
    """
    # get all value for single column from data matrix
    col_data = [row[col_num] for row in data]
    std_dev = statistics.stdev(col_data)
    return std_dev


def get_standardised_matrix(data: list) -> list:
    """
    standardize method removes the average and scales each column to unit
    variance. This operation is performed column-wise in an independent way.

    standardize method can be influenced by outliers(if they exist in
    the dataset) since it involves the estimation of the empirical average and
    standard deviation of each column.

    """
    data_standardization = []
    col_len = len(data[0])
    # get standardize data of column
    for col in range(col_len):
        col_data = [row[col] for row in data]
        N = len(col_data)
        average = sum(col_data) / N
        std_dev = get_standard_deviation(data, col)
        data_standardization.append(
            [(value - average) / std_dev for value in col_data]
            )
    # convert standardize data in standardised matrix
    standardised_matrix = [
        [row[i] for row in data_standardization]
        for i in range(len(data_standardization[0]))
    ]
    return standardised_matrix


def get_k_nearest_labels(
        list_data: list,
        learning_data: list, learning_data_labels: list, k: int) -> list:
    """
    find the k rows of the matrix learning_data that are the closest
    to the original data. and find k rows which is related in the matrix
    learning data labels.
    """
    distances = []
    # calculate distance between original data list and all learning data list
    for x in range(len(learning_data)):
        dist = get_distance(list_data, learning_data[x])
        distances.append((learning_data[x], dist, x))
    # sort distances ascending order
    distances.sort(key=operator.itemgetter(1))
    # get related k row in learning data labels
    k_nearest_labels = [
        learning_data_labels[distances[x][2]] for x in range(k)
        ]
    return k_nearest_labels


def get_mode(k_nearest_labels: list) -> int:
    """
     this function calculate mode for k_nearest_labels

    The mode of a sequence of numbers is the number with the highest frequency
    
    """
    k_nearest_labels = [j for i in k_nearest_labels for j in i]
    k_nearest_labels = multimode(k_nearest_labels)
    return random.choice(k_nearest_labels)
    # return statistics.mode(k_nearest_labels)


def classify(standardised_data: list,
             standardised_learning_data: list,
             learning_data_labels: list, k: int) -> list:
    """
    this function use 'get_k_nearest_labels' method and 'get_mode' method
    to predict the new label using learning data

    """
    data_labels = []
    # get new labels for data using learning data
    for row in standardised_data:
        k_nearest_labels = get_k_nearest_labels(
            row, standardised_learning_data, learning_data_labels, k
        )
        mode = get_mode(k_nearest_labels)
        data_labels.append(mode)
    return data_labels


def get_accuracy(correct_data_labels: list, data_labels: list) -> float:
    """
    This function calculate and return the percentage of accuracy. If both
    matrixes have exactly the same values (in exactly the same rownumbers)
    then the accuracy is of 100%. If only half of the values of both tables
    match exactly in terms of value and row number, then the accuracy is of 50%

    """
    count = 0
    N = len(correct_data_labels)
    # convert list of lists data to list.
    correct_data_labels = [j for i in correct_data_labels for j in i]
    for i in range(N):
        if correct_data_labels[i] == data_labels[i]:
            count += 1
        else:
            continue
    accuracy = (count / N) * 100
    return accuracy


def run_test():
    """
    This function create one matrix for each of these: correct_data_labels,
    learning_data and learning_data_labels (using load_from_csv). It
    standardise the matrix data and the matrix learning_data (using
    get_standardised_matrix). Then, it run the algorithm (using classify) and
    calculate the accuracy (using get_acuracy) for a series of experiments.
    In each experiment it run the algorithm (and calculate the accuracy)
    for different values of k (go from 3 to 15 in steps of 1), and show the
    results on the screen. For instance,
    if with k = 3 the accuracy is 68.5% it should show:
    k=3, Accuracy = 68.5%
    """
    data = load_from_csv("path of original dataset")
    correct_data_labels = load_from_csv("path of correct data labels dataset")
    learning_data = load_from_csv("path of learning dataset")
    learning_data_labels = load_from_csv("path of learning data labels dataset")
    

    # get standardised matrix for data matrix
    standardised_data = get_standardised_matrix(data)
    # get standardised matrix for learning data matrix
    standardised_learning_data = get_standardised_matrix(learning_data)
    # calculate accuracy for different values of k.
    for k in range(3, 16):
        data_labels = classify(
            standardised_data,
            standardised_learning_data,
            learning_data_labels,
            k,
        )
        accuracy = get_accuracy(correct_data_labels, data_labels)
        print(f"k={k}, Accuracy={accuracy:.2f}%")


if __name__ == "__main__":
    run_test()
