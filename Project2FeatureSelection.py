import numpy as np
import time
import math

def main():
    filename = input("Type in the name of the file to test: ")

    algorithm = select_algorithm()
    if algorithm == "1":
        forward_selection(read_in_data(filename))
    if algorithm == "2":
       backwards_elimination(read_in_data(filename))
    return

def select_algorithm():
    algorithm = input("Please decide the algorithm you want to use: " + '\n' +
                      "1. Forward Selection" + '\n' +
                      "2. Backwards Elimination" + '\n' 
                      )  
    return algorithm

def forward_selection(data):
    current_features = []
    best_accuracy_global = 0.0
    best_feature_set = [] 
    starting_time = time.time()
    for i in range(1, len(data[0])):
        adding_feature = 0
        best_accuracy_local = 0.0
        for k in range(1, len(data[0])):
            if k not in current_features:
                print("--Considering adding the", k, "feature ")
                accuracy = find_accuracy(current_features, data, k, 1)
                if accuracy > best_accuracy_local:
                    best_accuracy_local = accuracy
                    adding_feature = k
        current_features.append(adding_feature) 
        print("On level ", i, " I add the feature ", adding_feature, " to the current set")
        print("With ", len(current_features), " features, the accuracy is: ", best_accuracy_local * 100, "%")
        if best_accuracy_global >= best_accuracy_global: 
            best_accuracy_global = best_accuracy_local
            best_feature_set = list(current_features)
    ending_time = time.time()
    print("Set of features used: ", best_feature_set, "at accuracy: ", best_accuracy_global * 100, '\n', "Time passed: ", ending_time - starting_time)
    return

def backwards_elimination(data):
    best_accuracy_global = 0.0
    best_feature_set = []
    current_features = [i for i in range(1, len(data[0]))]
    starting_time = time.time()
    for i in range(1, len(data[0]) - 1):
        feature_pop = 0
        local_best_acc = 0.0
        for k in range(1, len(data[0]) - 1):
            if k in current_features:
                print("--Considering adding feature ", k)
                acc = find_accuracy(current_features, data, k, 2)
                if acc > local_best_acc:
                    local_best_acc = acc
                    feature_pop = k
        if feature_pop in current_features: 
            current_features.remove(feature_pop)
            print("On the level ", i, " the feature ", feature_pop, " is removed from the feauture set")
            print("With ", len(current_features), " features, the accuracy is: ", local_best_acc * 100, "%")
        if local_best_acc >= best_accuracy_global:
            best_accuracy_global = local_best_acc
            best_feature_set = list(current_features)
    ending_time = time.time()
    print("Set of features used: ", best_feature_set, "At accuracy: ", best_accuracy_global * 100, '\n', "Elapsed time: ", ending_time - starting_time)
    return

def read_in_data(filename):
    return np.loadtxt(filename)

def find_accuracy(features_set, data, feature_test, algorithm):
    testing_feat = list(features_set)
    if algorithm == 1:
        testing_feat.append(feature_test)
    if algorithm == 2:
        testing_feat.remove(feature_test)
    num_class = 0
    short_distance = math.inf
    result = 0
    for i in data:
        short_distance = math.inf
        for h in data:
            if not np.array_equal(h, i):
                distance = 0
                for j in testing_feat:
                    distance += pow((i[j] - h[j]), 2.0)
                if math.sqrt(distance) < short_distance:
                    short_distance = math.sqrt(distance)
                    result = h[0] 
        if result == i[0]:
            num_class += 1
    return num_class / (len(data) - 1)

main()
