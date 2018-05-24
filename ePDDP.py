"""
This module contains ePDDP, which tries to create even split.

For cluster selection criteria, it chooses the cluster with the most documents. For splitting criteria, it split the
cluster selected so that the two sub-clusters after splitting are of equal size.

This module also includes a main method to use the pddp algorithm directly. The command line argument used to run the
main method is:

python ePDDP.py input_file output_file representation number_clusters
"""

import numpy as np
from scipy.sparse import csc_matrix, linalg
from splitting_criteria import even_split
from selection_criteria import even
from helpers import get_scatter, read_in_data, matrix_comp, transformation, output_clustering_results
from PDDPNode import PDDPNode
import sys


def epddp(ms_raw, n_clusters, representation='tfidf'):
    """
    Method to perform ePDDP.

    :param ms_raw: input csc matrix to perform ePDDP algorithm
    :param n_clusters: desired number of clusters
    :param representation: representation of the data matrix. Should be either tfidf or norm_freq
    :return leafs: clusters created by PDDP algorithm
    :return root: the root of the PDDP tree
    """
    n, m = ms_raw.shape

    norm_ms = transformation(ms_raw, representation)

    # Check whether the transformation success or not
    if norm_ms is 0:
        return norm_ms

    ms = csc_matrix(norm_ms)

    # Before we start to split, we want to compute the fields of root node
    _, w = get_scatter(ms)

    root = PDDPNode(indices=np.arange(m), centroid=w,
                    leftvec=[], rightvec=[], scat=m)

    leafs = [root]

    # Main loop to iteratively split
    for i in range(1, n_clusters):
        # Here max_scat is used to store the number of documents in the node
        max_scat, max_node = even(leafs)

        # Now split the node with the largest size
        node_matrix = ms[:, max_node.indices]
        n_node, m_node = node_matrix.shape

        node_operator = matrix_comp(node_matrix, n_node, m_node)
        u, s, vt = linalg.svds(node_operator, k=1, which='LM')

        # Repack the fields of the divided node
        max_node.leftvec = u[:, 0]
        max_node.rightvec = vt[0]
        max_node.sval = s[0]

        left_indices, right_indices = even_split(max_node)

        left_matrix = ms[:, left_indices]
        _, w_left = get_scatter(left_matrix)

        right_matrix = ms[:, right_indices]
        _, w_right = get_scatter(right_matrix)

        max_node.leftchild = PDDPNode(indices=left_indices, centroid=w_left, leftvec=[], rightvec=[], scat=-1)
        max_node.rightchild = PDDPNode(indices=right_indices, centroid=w_right, leftvec=[], rightvec=[], scat=-1)

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

    clusters, root = epddp(ms_raw, N_clusters)

    print('Finish clustering')

    if clusters == 0:
        print('Because of invalid argument, algorithm exit without working')
    else:
        output_clustering_results(clusters, output_file)
