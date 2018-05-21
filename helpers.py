"""Helper methods used by other files"""
import numpy as np
from scipy.sparse import csc_matrix, linalg
from sklearn.feature_extraction.text import TfidfTransformer
import os

from PDDPNode import PDDPNode


def get_scatter(cols):
    """
    Compute the scatter value of a matrix.

    :param cols: the data matrix in sparse matrix representation
    :return scat: the scatter value of the input matrix
    :return meanv: the mean vector of the input matrix
    """
    n, p = cols.shape

    # Deal with the special cases for number of documents
    if p == 0:
        scat = 0
        meanv = []
        return scat, meanv
    elif p == 1:
        scat = 0
        meanv = cols
        return scat, meanv

    # Compute the mean vector of the cluster
    meanv = cols.mean(axis=1)

    # If difference along every row vary by less than 1e-12, then scatter = 0
    e = 0.000000000001
    row_min = cols.min(axis=1).toarray()
    row_max = cols.max(axis=1).toarray()

    min_norm = np.linalg.norm(row_min, np.inf)
    max_norm = np.linalg.norm(row_max, np.inf)
    if np.linalg.norm(row_max - row_min, ord=np.inf) > e * (1 + min_norm + max_norm):
        scat = linalg.norm(cols, ord='fro') ** 2 - 2 * \
               cols.T.dot(meanv).sum() + p * (np.linalg.norm(meanv, ord=2)) ** 2
    else:
        scat = 0
    return scat, meanv


def get_scatter_dense(m):
    """
    Get the scatter value and mean vector for dense matrix

    :param m: matrix to get scatter value and mean vector
    :return scat: scatter value of the matrix
    :return meanv: the mean vector of the matrix
    """
    n, p = m.shape

    # Deal with the special cases for number of documents
    if p == 0:
        scat = 0
        meanv = []
        return scat, meanv
    elif p == 1:
        scat = 0
        meanv = m
        return scat, meanv

    # Compute the mean vector of the cluster
    meanv = np.mean(m, axis=1)

    # if difference along every row vary by less than 1e-12, then scatter = 0
    e = 0.000000000001
    row_min = m.min(axis=1)
    row_max = m.max(axis=1)

    min_norm = np.linalg.norm(row_min, np.inf)
    max_norm = np.linalg.norm(row_max, np.inf)
    if np.linalg.norm(row_max - row_min, ord=np.inf) > e * (1 + min_norm + max_norm):
        scat = np.linalg.norm(m, ord='fro')
    else:
        scat = 0
    return scat, meanv


def read_in_data(input_data, col_based=True):
    """
    Read data from given filename and transform into sparse matrix

    This method assumes that each raw of the input data uses following format:

    row_id column_id word_count

    and the row_id and column_id starts from 1 (the returned matrix will correctly start from 0)
    :param input_data: the file name of the file that contains the input data
    :param col_based: whether each doc is stored in a column (True) or row (False)
    :return ms_raw: the sparse matrix extracted from the given file
    """
    with open(input_data, "r") as f_input:
        row = []
        col = []
        data = []
        m = 0
        n = 0
        for line in f_input:
            if col_based:
                c_i, r_i, d_i = line.split(" ")
            else:
                r_i, c_i, d_i = line.split(" ")
            row.append(int(r_i) - 1)
            col.append(int(c_i) - 1)
            data.append(float(d_i))
            if int(r_i) > n:
                n = int(r_i)
            if int(c_i) > m:
                m = int(c_i)

    row = np.asarray(row)
    col = np.asarray(col)
    data = np.asarray(data)
    # Transform the sparse representation into a sparse matrix
    # Notice that, each column represents a document and each row is a word
    ms_raw = csc_matrix((data, (row, col)), shape=(n, m))

    return ms_raw


def matrix_comp(A, n, m):
    """
    Define linear operator for a given matrix to be used in SVD.

    :param A: The matrix that is to be used for SVD
    :param n: number of dimensions of the input matrix
    :param m: number of documents in the current cluster
    :return node_operator: the linear operator needed to conduct SVD on this node
    """
    node_mean = A.mean(axis=1)

    def mv(v):
        """Define A * v for svd"""
        return A.dot(v).reshape((n, 1)) - node_mean * np.dot(np.ones(m).reshape((1, m)), v)

    def rmv(v):
        """Define A^H * v for svd, where A^H is the conjugate transpose of A"""
        return A.T.dot(v).reshape((m, 1)) - np.ones(m).reshape((m, 1)) * np.dot(v.T, node_mean)

    def mm(V):
        """Define A * V for svd, where V is a dense matrix. This is the normal matrix multiplication"""
        return A.dot(V)

    node_operator = linalg.LinearOperator(
        shape=(n, m), matvec=mv, rmatvec=rmv, matmat=mm)

    return node_operator


def transformation(ms_raw, representation, col_based=True):
    """
    Perform transformation on the input data matrix

    This method transforms raw input data matrix to either tf-idf or normed bag-of-words (norm frequency)
    :param ms_raw: input data matrix
    :param representation: representation required. Should be "tfidf" or "norm_freq"
    :param col_based: whether the input data matrix is column-based (each column represents a document) or row-based
    (each row represents a document)
    :return: transformed data matrix
    """
    if col_based:
        if representation == 'tfidf':
            transformer = TfidfTransformer()
            norm_ms = transformer.fit_transform(ms_raw.T).T
        elif representation == 'norm_freq':
            # Normalize each document to have length 1
            norm_ms = ms_raw / np.sqrt(ms_raw.power(2).sum(axis=0))
        else:
            print('The representation should be \"tfidf\" or \"norm_freq\"')
            return 0
    else:
        if representation == 'tfidf':
            transformer = TfidfTransformer()
            norm_ms = transformer.fit_transform(ms_raw)
        elif representation == 'norm_freq':
            # Normalize each document to have length 1
            norm_ms = ms_raw / np.sqrt(ms_raw.power(2).sum(axis=1))
        else:
            print('The representation should be \"tfidf\" or \"norm_freq\"')
            return 0

    return norm_ms


def output_clustering_results(clusters, output_file):
    """
    Output the clustering results to output file

    :param clusters: list of clustering results
    :param output_file: file to store the output results
    """
    if clusters == 0:
        print('Exist without clustering')
    else:
        with open(output_file, 'w') as f:
            # Write the clustering results into an output file
            for cluster in clusters:
                leading = cluster.indices[0]
                f.write(str(leading + 1) + ":")
                for doc in cluster.indices:
                    f.write(str(doc + 1) + '\t')
                f.write('\n')


# def choose_dataset(dataset, cwd, terms='stemmed'):
#     """
#     Assemble the input filename and output filename
#     :param terms: whether to use doc2vec, all terms or stemmed terms
#     :param dataset: dataset chosen
#     :param cwd: current working directory
#     :return: input filename and output filename
#     """
#     path = os.path.join(cwd, dataset)
#     if terms == 'doc2vec':
#         input_file = os.path.join(path, 'doc2vec.txt')
#     elif terms == 'doc2vec_all':
#         input_file = os.path.join(path, 'doc2vec_all.txt')
#     elif terms == 'all':
#         input_file = os.path.join(path, 'matrix_all.in')
#     else:
#         input_file = os.path.join(path, 'matrix.in')
#     output_file = os.path.join(path, 'results.txt')
#
#     return input_file, output_file


def compute_node(ms, indices):
    """
    Compute fields of node required
    :param ms: the original data matrix
    :param indices: indices of documents contained in the node
    :return node: the PDDPNode with needed fields
    """
    matrix = ms[:, indices]
    s, w = get_scatter(matrix)
    node = PDDPNode(indices=indices, centroid=w, leftvec=[], rightvec=[], scat=s)
    return node
