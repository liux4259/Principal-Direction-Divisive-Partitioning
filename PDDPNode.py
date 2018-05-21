"""Contains the definition to PDDPNode used in other files"""


class PDDPNode(object):
    """
    Defines PDDP_node to be used in PDDP algorithm.

    """

    def __init__(self, indices, centroid, leftvec, rightvec, scat=-1, leftchild=0, rightchild=0, sval=0, index=-1):
        """
        Initialize a PDDP node
        :param indices: Indices of documents in this node's cluster (list of p integers)
        :param centroid: The mean vector w of documents in cluster (n-vector)
        :param leftvec: Left singular vector u corresponding to sigma (n-vector)
        :param rightvec: Right sigular vector v corresponding to sigma (p-vector)
        :param scat: Total scatter value for this cluster (a scalar)
        :param leftchild: Pointer to the left child node (a PDDP_node)
        :param rightchild: Pointer to the right child node (a PDDP_node)
        :param sval: Leading sigular value sigma for this cluster (a scalar)
        :param index: the index used for splitting. Notice that this is not necessarily a real index. It can be anything
        that is used for splitting.
        """
        self.indices = indices
        self.centroid = centroid
        self.sval = sval
        self.leftvec = leftvec
        self.rightvec = rightvec
        self.scat = scat
        self.leftchild = leftchild
        self.rightchild = rightchild
        self.index = index
