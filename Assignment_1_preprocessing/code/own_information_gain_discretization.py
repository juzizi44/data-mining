import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math

'''信息增益离散化具体实现'''


def compute_midpoints(data, attribute):
    sorted_values = data[attribute].sort_values()
    return (sorted_values.values[1:] + sorted_values.values[:-1]) / 2


def compute_entropy(labels):
    _, counts = np.unique(labels, return_counts=True)
    probs = counts / len(labels)
    return -np.sum(probs * np.log2(probs))


def split_data(data, attribute, midpoint):
    left = data[data[attribute] <= midpoint]
    right = data[data[attribute] > midpoint]
    return left, right


def discretize(subset, attribute, depth, threshold, max_depth):
    if len(subset['class'].unique()) == 1 or len(subset) < threshold or depth == max_depth:
        return [subset]
    else:
        midpoints = compute_midpoints(subset, attribute)
        entropies = []
        for midpoint in midpoints:
            left, right = split_data(subset, attribute, midpoint)
            left_entropy = compute_entropy(left['class'].values)
            right_entropy = compute_entropy(right['class'].values)
            midpoint_entropy = (len(left) * left_entropy +
                                len(right) * right_entropy) / len(subset)
            entropies.append(midpoint_entropy)
        best_index = np.argmin(entropies)
        best_midpoint = midpoints[best_index]
        left, right = split_data(subset, attribute, best_midpoint)
        return discretize(left, attribute, depth + 1, threshold, max_depth) + discretize(right, attribute, depth + 1,
                                                                                         threshold, max_depth)
