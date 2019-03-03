from collections import Counter
from linear_algebra import distance
from data import cities
from statistics import mean
import math, random
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
import numpy as np


def majority_vote(labels):
    """assumes that labels are ordered from nearest to farthest"""
    vote_counts = Counter(labels)
    winner, winner_count = vote_counts.most_common(1)[0]
    num_winners = len([count
                       for count in vote_counts.values()
                       if count == winner_count])

    if num_winners == 1:
        return winner                     # unique winner, so return it
    else:
        return majority_vote(labels[:-1]) # try again without the farthest


def knn_classify(k, labeled_points, new_point):
    """each labeled point should be a pair (point, label)"""

    # order the labeled points from nearest to farthest
    by_distance = sorted(labeled_points,
                         key=lambda point_label: distance(point_label[0], new_point))

    # find the labels for the k closest
    k_nearest_labels = [label for _, label in by_distance[:k]]

    # and let them vote
    return majority_vote(k_nearest_labels)


def predict_preferred_language_by_city(k_values, cities):
    """
    TODO
    predicts a preferred programming language for each city using above knn_classify() and
    counts if predicted language matches the actual language.
    Finally, print number of correct for each k value using this:
    print(k, "neighbor[s]:", num_correct, "correct out of", len(cities))
    """
    for i in k_values:

        num_correct = 0
        for j, z in enumerate(cities):
            new_list=cities.copy()
            new_list.remove(z)
            predicted_language=knn_classify(i,new_list,z[0])
            if (predicted_language== cities[j][1]):
                num_correct = num_correct + 1
        print(i, "neighbor[s]:", num_correct, "correct out of", len(cities))

def get_i_label(cities):
    i= []
    label = []
    for city in cities:
        i.append([city[0][0],city[0][1]])
        label.append(city[1])

    return np.array(i),np.array(label)

def predict_preferred_language_by_city_scikit(k_values, cities):
    print("Output after using KNN by scikit")
    for k in k_values:
        num_correct = 0
        knn = KNeighborsClassifier(n_neighbors=k)
        for city in cities:
            new_list = cities.copy()
            new_list.remove(city)
            i,label = get_i_label(new_list)
            knn.fit(i,label)
            predicted_language = knn.predict(np.array([[city[0][0],city[0][1]]]))

            if predicted_language == city[1]:
                num_correct += 1

        print(k,"neighbor[s]:", num_correct, "correct out of", len(cities))

if __name__ == "__main__":

    k_values = [1, 3, 5, 7]
    # TODO
    # Import cities from data.py and pass it into predict_preferred_language_by_city(x, y).

    predict_preferred_language_by_city(k_values, cities)
    predict_preferred_language_by_city_scikit(k_values,cities)



