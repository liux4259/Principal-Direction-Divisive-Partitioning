"""Contains splitting criterion"""
import numpy as np
from nltk.cluster import KMeansClusterer, euclidean_distance
from scipy.sparse import linalg


def scatter_split(max_node, ms, method='scatter_project'):
    """
    Split the cluster to minimize the total scatter value after split

    :param method: whether split using scatter in original space or projected space
    :param max_node: the PDDP node to split next
    :param ms: the original data matrix
    :return left_indices: indices of documents to left node
    :return right_indices: indices of documents to right node
    """
    if method == 'scatter':
        min_scat, min_index, sorted_tup = _find_cutoff_scatter(max_node, ms)
    elif method == 'scatter_project':
        min_scat, min_index, sorted_tup = _find_cutoff_scatter_project(max_node)
    elif method == 'scatter_dense':
        min_scat, min_index, sorted_tup = _find_cutoff_scatter_dense(max_node, ms)
    else:
        print('Invalid method name in scatter_split')
        return
    left_indices = [tup[0] for i, tup in enumerate(sorted_tup) if i <= min_index]
    right_indices = [tup[0] for i, tup in enumerate(sorted_tup) if i > min_index]

    return left_indices, right_indices


def gap(max_node, tau=0.4):
    """
    Split the cluster at the maximum gap

    :param max_node: the PDDP node to split next
    :param tau: the percentage to ignore at each end
    :return left_indices: indices of documents to left node
    :return right_indices: indices of documents to right node
    """
    max_index, max_gap, sorted_tup = find_cutoff_gap(max_node, tau)

    left_indices = [sorted_tup[j][0] for j in range(len(sorted_tup)) if j <= max_index]
    right_indices = [sorted_tup[j][0] for j in range(len(sorted_tup)) if j > max_index]

    return left_indices, right_indices


def sparse_cut(max_node):
    """
    Split the cluster at the most sparse area

    :param max_node: the PDDP node to split next
    :return left_indices: indices of documents to left node
    :return right_indices: indices of documents to right node
    """
    max_rightval = max_node.index

    p_indices = max_node.indices
    p_rightvec = max_node.rightvec

    left_indices = [p_indices[j] for j, ele in enumerate(p_rightvec) if ele >= max_rightval]
    right_indices = [p_indices[j] for j, ele in enumerate(p_rightvec) if ele < max_rightval]

    return left_indices, right_indices


def origin(max_node, frac=0):
    """
    Split based on the value of right singular vector, as in original PDDP

    :param max_node: the PDDP node to be split next
    :param frac: used to control the skewness of splitting. If the fraction of documents into one of the cluster is
    smaller than frac, split from the median of the value of right singular value instead of 0
    :return left_indices: indices of documents to left node
    :return right_indices: indices of documents to right node
    """
    p_indices = max_node.indices
    left_indices = [p_indices[j] for j, ele in enumerate(max_node.rightvec) if ele >= 0]
    right_indices = [p_indices[j] for j, ele in enumerate(max_node.rightvec) if ele < 0]

    # If the split is too skew, re-split using the median of left singular vector as threshold.
    threshold = frac * len(max_node.indices)
    if len(right_indices) < threshold or len(left_indices) < threshold:
        left_indices, right_indices = even_split(max_node)
    return left_indices, right_indices


def kmeans(max_node):
    """
    Split using kmenas on right singular vector

    :param max_node: the PDDP node to be split next
    :return left_indices: indices of documents to left node
    :return right_indices: indices of documents to right node
    """
    indices = max_node.indices
    v = max_node.rightvec.reshape(-1, 1)

    clusterer = KMeansClusterer(2, euclidean_distance)
    kmeans_result = clusterer.cluster(v, True, trace=False)

    left_indices = [indices[i] for i, label in enumerate(kmeans_result) if label == 0]
    right_indices = [indices[i] for i, label in enumerate(kmeans_result) if label == 1]
    return left_indices, right_indices


def even_split(max_node):
    """
    Evenly split the cluster based on their right singular value

    :param max_node: the PDDP node to be split next
    :return left_indices: indices of documents to left node
    :return right_indices: indices of documents to right node
    """
    p_indices = max_node.indices
    vt_copy = np.copy(max_node.rightvec)
    vt_copy = np.sort(vt_copy)
    median = vt_copy[len(vt_copy) // 2]

    left_indices = [p_indices[i] for i, ele in enumerate(max_node.rightvec) if ele >= median]
    right_indices = [p_indices[i] for i, ele in enumerate(max_node.rightvec) if ele < median]
    return left_indices, right_indices


def find_cutoff_gap(node, tau=0.15):
    """
    Find the largest gap from a given cluster

    :param node: the PDDP node to find largest gap from
    :param tau: ignored portion of each tail
    :return max_index: the index with the largest gap. Notice that this is not the index of the original array. Rather,
    it is the index of the sorted array
    :return max_gap: maximum gap in the cluster
    :return sorted_tup: the list of tuples returned from sort_tuple
    """
    sorted_tup = _sort_tuple(node)

    # Find gaps between each element in the sorted list
    max_gap = -np.inf
    max_index = -1
    start = int(np.floor(len(sorted_tup) * tau))
    end = len(sorted_tup) - start
    for j in range(start, end - 1):
        current_gap = sorted_tup[j + 1][1] - sorted_tup[j][1]
        if current_gap > max_gap:
            max_gap = current_gap
            max_index = j

    return max_index, max_gap, sorted_tup


def _find_cutoff_scatter_project(node):
    """
    Find the split point that minimize total scatter value in projected space.

    To increase efficiency, we calculate the scatter value by keep track of the sum (equivalent to mean) and sum of
    squares of the right singular values.
    :param node: the PDDP node to split
    :return: min_index: the index with the minimum scatter value.
    :return: min_scatter: minimum total scatter value after split
    :return: sorted_tup: the list of tuples returned from sort_tuple
    """
    sorted_tup = _sort_tuple(node)

    # Use sum instead of mean to reduce computation needed
    left_sum = 0
    right_sum = node.rightvec.sum()

    # Initialize norm value. It is the sum of all right singular values in the sub-cluster
    right_norm = np.linalg.norm(node.rightvec)
    left_norm = 0

    min_scat = np.inf
    min_index = -1

    for i, tup in enumerate(sorted_tup):
        if i == len(sorted_tup) - 1:
            continue

        r_val = tup[1]

        # Update sums
        left_sum += r_val
        right_sum -= r_val

        # Update sum of squares
        r_norm = r_val ** 2
        left_norm += r_norm
        right_norm -= r_norm

        # Calculate scatter values
        left_scatter = left_norm - left_sum ** 2 / (i + 1)
        right_scatter = right_norm - right_sum ** 2 / (len(sorted_tup) - i - 1)
        total_scatter = left_scatter + right_scatter

        if total_scatter < min_scat:
            min_scat = total_scatter
            min_index = i

    return min_scat, min_index, sorted_tup


def _find_cutoff_scatter(node, ms):
    """
    Find the split point that minimize total scatter value in original space after split

    To increase efficiency, we keep track of the mean vectors and the Frobenius norm of each sub-cluster and use them to
    calculate the scatter value in the original space.
    :param node: the PDDP node to be split
    :param ms: the original data matrix
    :return: min_index: the index with the minimum scatter value
    :return: min_scatter: minimum total scatter value in original space after split
    :return: sorted_tup: the list of tuples returned from sort_tuple
    """
    sorted_tup = _sort_tuple(node)

    m = len(node.indices)
    n = ms.shape[0]

    cluster_matrix = ms[:, node.indices]

    # Use sum instead of mean vectors to reduce useless computations
    left_sum = np.zeros([n, 1])
    right_sum = cluster_matrix.sum(axis=1)

    # Build right_indices with the reverse order as sorted_tup for efficient pop
    left_indices = []
    right_indices = []

    for i in range(len(sorted_tup)):
        tup = sorted_tup[len(sorted_tup) - i - 1]
        right_indices.append(tup[0])

    # Initialize norms
    right_fro_norm = linalg.norm(cluster_matrix, ord='fro')
    left_fro_norm = 0

    min_scat = np.inf
    min_index = -1

    for i in range(m - 1):
        next_doc = right_indices.pop()
        left_indices.append(next_doc)
        vec = ms.getcol(next_doc)

        # Update sum vectors
        left_sum += vec
        right_sum -= vec

        # Update fro_norms
        vec_norm = linalg.norm(vec)
        left_fro_norm = left_fro_norm + vec_norm
        right_fro_norm = right_fro_norm - vec_norm

        # Update scatter values
        left_scatter = left_fro_norm - np.dot(left_sum.T, left_sum) / (i + 1)
        right_scatter = right_fro_norm - np.dot(right_sum.T, right_sum) / (m - i - 1)
        total_scatter = left_scatter + right_scatter

        if total_scatter < min_scat:
            min_scat = total_scatter
            min_index = i
    return min_scat, min_index, sorted_tup


def _find_cutoff_scatter_dense(node, ms):
    """
    Dense version of find_cutoff_scatter.

    :param node: node to split
    :param ms: original data matrix
    :return min_index: the index with the minimum scatter value
    :return min_scatter: minimum total scatter value after split
    :return sorted_tup: the list of tuples returned from sort_tuple
    """
    sorted_tup = _sort_tuple(node)

    m = len(node.indices)
    n = ms.shape[0]

    cluster_matrix = ms[:, node.indices]

    # Use sum instead of mean vectors to reduce useless computations
    left_sum = np.zeros([n, 1])
    right_sum = np.sum(cluster_matrix, axis=1).reshape(n, 1)

    # Build right_indices with the reverse order as sorted_tup for efficient pop
    left_indices = []
    right_indices = []

    for i in range(len(sorted_tup)):
        tup = sorted_tup[len(sorted_tup) - i - 1]
        right_indices.append(tup[0])

    # Initialize norms
    right_fro_norm = np.linalg.norm(cluster_matrix, ord='fro')
    left_fro_norm = 0

    min_scat = np.inf
    min_index = -1

    for i in range(m - 1):
        next_doc = right_indices.pop()
        left_indices.append(next_doc)

        vec = ms[:, next_doc].reshape(n, 1)

        # Update sum vectors
        left_sum += vec
        right_sum -= vec

        # Update fro_norm
        vec_norm = np.linalg.norm(vec)
        left_fro_norm = left_fro_norm + vec_norm
        right_fro_norm = right_fro_norm - vec_norm

        # Update scatter values
        left_scatter = left_fro_norm - np.dot(left_sum.T, left_sum) / (i + 1)
        right_scatter = right_fro_norm - np.dot(right_sum.T, right_sum) / (m - i - 1)
        total_scatter = left_scatter + right_scatter

        if total_scatter < min_scat:
            min_scat = total_scatter
            min_index = i

    return min_scat, min_index, sorted_tup


def _sort_tuple(node):
    """
    Create a tuple and sort it according to the right singular value

    The first value of the tuple is the index of the doc
    The second value of the tuple is the right singular value
    :param node: The PDDp node to sort
    :return sorted_tup: the sorted list of tuples sorted by right singular value
    """
    v_tuple = []
    for j in range(len(node.indices)):
        indices = node.indices[j]
        right_val = node.rightvec[j]
        temp_tup = (indices, right_val)
        v_tuple.append(temp_tup)

    sorted_tup = sorted(v_tuple, key=lambda tup: tup[1])

    return sorted_tup


def find_cutoff_sparse(node):
    """
    Find the most sparse area in a cluster

    :param node: the PDDP node to find sparse area
    :return max_sparse: max sparsity value
    :return max_rightval: the right singular value, which is the projection, corresponds to max sparsity value
    :return max_index: the index of document corresponding to max sparsity
    """
    max_sparse = -np.inf
    max_rightval = -1
    right_vec = node.rightvec
    for j in range(len(node.indices)):
        sparse = _density(right_vec[j], right_vec)
        if sparse > max_sparse:
            max_sparse = sparse
            max_rightval = right_vec[j]

    return max_sparse, max_rightval


def _multivariate_kernel(x):
    """
    Calculate the multivariate kernel for 1 dimension input

    :param x: input value to compute multivariate kernel
    :return: the multivariate kernel of the input
    """
    return np.exp(-0.5 * np.absolute(x)) / np.sqrt(2 * np.pi)


def _density(x, v, h=1.5):
    """
    Compute the density estimation for a given 1 dimension input

    :param h: size of bucket
    :param x: input to compute density
    :param v: right singular vectors returned from svd
    :return: the multivariate kernel density estimator of the given input
    """
    kernel = _multivariate_kernel((x - v) / h)
    return np.sum(kernel) / (len(v) * h)
