import heapq
import math
import numpy as np
from operator import itemgetter


class LSH:

    Name = "LSH"

    def __init__(self):
        self._m = None  # Dimensionality
        self._n_H = None  # Number of hash functions
        self._n_g = None  # Number of groups of hash functions
        self._f_H = None  # Hash functions to be used
        self._H = None  # Hash tables used for searching
        self._Q = None  # Item vectors
        self._G_m = None  # Multi-dimensional normal mean
        self._G_v = None  # Multi-dimensional normal variance

    def create_index(self, Q, n, m, nh, ng):
        self._m = m
        self._Q = Q
        self._n_H = nh
        self._n_g = ng
        self._G_m = np.array([0] * m)
        self._G_v = np.eye(m)
        self._f_H = []

        for gi in xrange(ng):
            g = []
            for fi in xrange(nh):
                g.append(np.random.multivariate_normal(self._G_m, self._G_v))
            self._f_H.append(g)
        
        self._H = []
        for gi in xrange(ng):
            self._H.append({})

        for ti in xrange(n):
            for gi in xrange(ng):
                h_key = ""
                for f in self._f_H[gi]:
                    if Q[ti].dot(f) >= 0:
                        h_key = h_key + "1"
                    else:
                        h_key = h_key + "0"

                if h_key in self._H[gi]:
                    self._H[gi][h_key].append(ti)
                else:
                    self._H[gi][h_key] = [ti]

    def top_item(self, u, k):
        heap_r = []  # a priority queue for candidate results

        for gi in xrange(ng):
            h_key = ""
            for f in self._f_H[gi]:
                if u.dot(f) >= 0:
                    h_key = h_key + "1"
                else:
                    h_key = h_key + "0"
            #print h_key

            if h_key in self._H[gi]:
                for tid in self._H[gi][h_key]:
                    value = self._Q[tid].dot(u)
                    if (len(heap_r) < k):
                        heapq.heappush(heap_r, (value, tid))
                    else:
                        if (value > heap_r[0][0]):
                            heapq.heappushpop(heap_r, (value, tid))
                break
            # else:
            #     print "Cannot find corresponding bucket!"

        res = []
        for r in heap_r:
            res.append(r[1])
        return res


if __name__ == "__main__":
    """
    Unit test procedure
    """
    T = LSH()
    m = 10
    n = 10000
    nh = 3
    ng = 5
    Q = np.random.random_sample([n, m])
    # print Q

    T.create_index(Q, n, m, nh, ng)
    # T.print_index()

    u = np.array([0.1, -0.1, 0.1, -0.1, 0.1, -0.1, 0.1, -0.1, 0.1, -0.1])
    maxval = -1
    for i in xrange(n):
        maxval = max(maxval, Q[i].dot(u))
    print maxval

    res = T.top_item(u, 1)
    #print Q[res[0]]
    print Q[res[0]].dot(u)
