"""
Module to perform PDDP algorithm.

This module includes the PDDP variants that uses scatter value as cluster selection criteria and different splitting
criteria. This module uses sparse data representation from scipy.

This module also includes a main method to use the pddp algorithm directly. The command line argument used to run the
main method is:

python PDDP.py input_file output_file splitting_criteria representation number_clusters
"""

import numpy as np
from scipy.sparse import csc_matrix, linalg
from splitting_criteria import origin, gap, kmeans, scatter_split, even_split
from selection_criteria import scatter
from helpers import compute_node, read_in_data, matrix_comp, transformation, output_clustering_results
import sys


def pddp(ms_raw, n_clusters, representation='tfidf', split='origin', tol=0.01):
    """
    Method to perform PDDP algorithm that uses scatter value as cluster selection criteria

    :param ms_raw: input csc matrix to perform PDDP algorithm
    :param n_clusters: desired number of clusters
    :param representation: representation of the data matrix. Should be either tfidf or norm_freq
    :param split: splitting criteria. Available options: gap, origin, kmeans, scatter, scatter_project, even
    :param tol: tolerance of scatter value to stop splitting
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
    root = compute_node(ms, np.arange(m))

    # Current clusters are stored in a list called "leafs", which should be returned
    leafs = [root]

    # Main loop to iteratively split
    for i in range(1, n_clusters):
        max_scat, max_node = scatter(leafs)

        print(max_scat)

        if max_scat < tol:
            print('\nbreak with {} clusters\n'.format(i - 1))
            break

        # Now split the node with largest scatter value
        node_matrix = ms[:, max_node.indices]
        n_node, m_node = node_matrix.shape

        node_operator = matrix_comp(node_matrix, n_node, m_node)
        u, s, vt = linalg.svds(node_operator, k=1, which='LM')

        # Repack the fields of the divided node
        max_node.leftvec = u[:, 0]
        max_node.rightvec = vt[0]
        max_node.sval = s[0]

        # Check which splitting criteria to use
        if split == 'gap':
            left_indices, right_indices = gap(max_node, tau=0.15)
        elif split == 'origin':
            left_indices, right_indices = origin(max_node)
        elif split == 'kmeans':
            left_indices, right_indices = kmeans(max_node)
        elif split == 'scatter':
            left_indices, right_indices = scatter_split(max_node=max_node, ms=ms, method='scatter')
        elif split == 'scatter_project':
            left_indices, right_indices = scatter_split(max_node=max_node, ms=ms, method='scatter_project')
        elif split == 'even':
            left_indices, right_indices = even_split(max_node)
        else:
            print('The argument split is invalid')
            return 0

        max_node.leftchild = compute_node(ms, left_indices)
        max_node.rightchild = compute_node(ms, right_indices)

        # After splitting, remove the split node from leaf list and add the left and right node to the leaf node
        leafs.remove(max_node)
        leafs.append(max_node.leftchild)
        leafs.append(max_node.rightchild)

    return leafs, root


if __name__ == '__main__':
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    splitting_criteria = sys.argv[3]
    representation = sys.argv[4]
    N_clusters = int(sys.argv[5])

    ms_raw = read_in_data(input_file)

    clusters, root = pddp(ms_raw, N_clusters, representation=representation, split=splitting_criteria)

    if clusters == 0:
        print('Because of invalid argument, algorithm exit without working')
    else:
        output_clustering_results(clusters, output_file)
