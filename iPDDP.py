"""Module to perform iPDDP algorithm.

For cluster selection criteria, it chooses the cluster with the largest gap in the node, where gap of a point is defined
as the distance between the point and the next consecutive point in the projected space. For splitting criteria, it
split at the point at the largest gap of the selected cluster, with gap as defined above

This module also includes a main method to use the pddp algorithm directly. The command line argument used to run the
main method is:

python iPDDP.py input_file output_file representation number_clusters
"""

import numpy as np
from scipy.sparse import csc_matrix, linalg
from splitting_criteria import gap
from selection_criteria import largest_gap
from helpers import get_scatter, read_in_data, matrix_comp, transformation, output_clustering_results
from PDDPNode import PDDPNode
import sys


def ipddp(ms_raw, n_clusters, representation='tfidf', min_frac=-1):
    """
    Method to perform iPDDP algorithm.

    :param ms_raw: input csc matrix to perform iPDDP algorithm
    :param n_clusters: desired number of clusters
    :param representation: representation of the data matrix. Should be either tfidf or norm_freq
    :param min_frac: minimum fraction of total number of clusters to be split further
    :return leafs: clusters created by PDDP algorithm
    :return root: the root of the PDDP tree
    """
    n, m = ms_raw.shape
    norm_ms = transformation(ms_raw, representation)

    # Check whether the transformation success or not
    if norm_ms is 0:
        return norm_ms

    ms = csc_matrix(norm_ms)

    # Checking parameters passed in
    if n_clusters > m:
        n_clusters = m

    # Before we start to split, we want to compute the fields of root node
    _, w = get_scatter(ms)

    node_operator = matrix_comp(ms, n, m)
    u, s, vt = linalg.svds(node_operator, k=1, which='LM')

    root = PDDPNode(indices=np.arange(m), centroid=w, leftvec=u[:, 0], rightvec=vt[0], sval=s[0])
    leafs = [root]

    if min_frac == -1:
        min_frac = n_clusters / m
    min_pts = np.floor(len(root.indices) * min_frac)

    # Main loop to iteratively divide the cluster
    for i in range(1, n_clusters):
        max_scat, max_node = largest_gap(leafs, min_pts)

        # print(max_scat)

        left_indices, right_indices = gap(max_node)

        left_matrix = ms[:, left_indices]
        s_left, w_left = get_scatter(left_matrix)

        right_matrix = ms[:, right_indices]
        s_right, w_right = get_scatter(right_matrix)

        # Now split the node with largest scatter value
        left_matrix = ms[:, left_indices]
        n_left, m_left = left_matrix.shape

        operator_left = matrix_comp(left_matrix, n_left, m_left)
        u_left, s_left, vt_left = linalg.svds(operator_left, k=1, which='LM')

        right_matrix = ms[:, right_indices]
        n_right, m_right = right_matrix.shape

        operator_right = matrix_comp(right_matrix, n_right, m_right)
        u_right, s_right, vt_right = linalg.svds(operator_right, k=1, which='LM')

        max_node.leftchild = PDDPNode(indices=left_indices, centroid=w_left, leftvec=u_left[:, 0], rightvec=vt_left[0],
                                      sval=s_left)
        max_node.rightchild = PDDPNode(indices=right_indices, centroid=w_right, leftvec=u_right[:, 0],
                                       rightvec=vt_right[0], sval=s_right)

        # After splitting, remove the split node from leaf list and add the left and right node to the leaf node
        leafs.remove(max_node)
        leafs.append(max_node.leftchild)
        leafs.append(max_node.rightchild)

    return leafs, root


if __name__ == '__main__':
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    splitting_criteria = sys.argv[3]
    N_clusters = int(sys.argv[4])

    ms_raw = read_in_data(input_file)

    clusters, root = ipddp(ms_raw, N_clusters)

    print('Finish clustering')

    if clusters == 0:
        print('Because of invalid argument, algorithm exit without working')
    else:
        output_clustering_results(clusters, output_file)
