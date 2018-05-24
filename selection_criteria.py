"""Module that contains different splitting criteria"""
import numpy as np
from splitting_criteria import find_cutoff_gap, find_cutoff_sparse


def scatter(leafs):
    """
    Find the max scatter value and node with max scatter value
    :param leafs: is the list contains the current leaf nodes
    :return max_scat: maximum scatter value in leafs
    :return max_node: node with maximum scatter value
    """
    max_scat = -float(np.inf)
    max_node = None
    for node in leafs:
        if node.scat > max_scat:
            max_scat = node.scat
            max_node = node
    return max_scat, max_node


def largest_gap(leafs, min_pts):
    """
    Find the cluster with largest gap

    :param leafs: the list of clusters to compare
    :param min_pts: minimum number of docs in a cluster to be split next.
    :return max_gap: the maximum gap found in the list of clusters
    :return max_node: node with largest gap
    """
    max_gap = -float(np.inf)
    max_node = None
    for node in leafs:
        # Here we use the field "scat" to store the gap because we will not use the real scatter value
        if node.scat == -1:
            # node.scat equals -1 means the scat field is not calculated for this node
            index, gap, _ = find_cutoff_gap(node)
            node.scat = gap
            node.index = index
        if node.scat > max_gap and len(node.indices) > min_pts:
            max_gap = node.scat
            max_node = node

    return max_gap, max_node


def sparse(leafs):
    """
    Find the cluster with the most sparse area

    :param leafs: the list of clusters to compare
    :return min_density: the density estimation of the most sparse area
    :return min_node: the node with lowest density estimation
    """
    min_density = float(np.inf)
    min_node = None
    for node in leafs:
        # Here we use field “scat" to store the density because we will not use the real scatter value
        if node.scat == -1:
            # node.scat equals -1 means the scat field is not calculated for this node
            sparsity, rightval = find_cutoff_sparse(node)
            node.scat = sparsity
            node.index = rightval
        if node.scat < min_density:
            min_density = node.scat
            min_node = node

    return min_density, min_node


def even(leafs):
    """
    Find the largest cluster in the leafs

    :param leafs: the list of clusters to compare
    :return max_size: the maximum size of cluster in the list
    :return max_node: the node with maximum size
    """
    max_size = -1
    max_node = None
    for node in leafs:
        if node.scat == -1:
            size = len(node.indices)
            node.scat = size
        if node.scat > max_size:
            max_size = node.scat
            max_node = node

    return max_size, max_node
