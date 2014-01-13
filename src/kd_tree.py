import heapq
import numpy as np
from operator import itemgetter


def upper_bound(lbound, rbound, u):
    ubound = 0
    for i in xrange(len(u)):
        if u[i] > 0:
            ubound += rbound[i] * u[i]
        else:
            ubound += lbound[i] * u[i]
    return ubound


class Node:

    Name = "Node"

    def __init__(self):
        self._lchild = None
        self._rchild = None
        self._lbound = None
        self._rbound = None
        self._count = 0
        self._tids = None


class KDTree:

    Name = "KDTree"

    def __init__(self):
        self._m = None
        self._n_root = None
        self._Q = None

    def create_index(self, Q, n, m):
        self._m = m
        self._Q = Q
        lbound = np.array([10000] * m)
        rbound = np.array([-10000] * m)
        for ti in xrange(n):
            lbound = np.minimum(Q[ti], lbound)
            rbound = np.maximum(Q[ti], rbound)
        self._n_root = self.build_tree(Q, range(n), 0, lbound, rbound, m)

    def build_tree(self, Q, tids, level, lbound, rbound, m):
        node = Node()
        node._lbound = list(lbound)
        node._rbound = list(rbound)
        node._count = len(tids)

        if (node._count <= 50):
            node._tids = tids
            return node

        att = level % m

        copy = [(tid, Q[tid][att]) for tid in tids]
        copy = sorted(copy, key=itemgetter(1))
        median = 0
        if node._count % 2 == 1:
            median = copy[(node._count - 1) / 2][1]
        else:
            median = (copy[node._count / 2 - 1][1] + copy[node._count / 2][1]) / 2

        lc_tids = []
        rc_tids = []
        for tid in tids:
            if (Q[tid][att] <= median):
                lc_tids.append(tid)
            else:
                rc_tids.append(tid)

        tmp_val = rbound[att]
        rbound[att] = median
        node._lchild = self.build_tree(Q, lc_tids, level + 1, lbound, rbound, m)
        rbound[att] = tmp_val
        lbound[att] = median
        node._rchild = self.build_tree(Q, rc_tids, level + 1, lbound, rbound, m)

        return node

    def top_item(self, u, k):
        heap_n = []  # a priority queue for index nodes
        heap_r = []  # a priority queue for candidate results

        n = self._n_root
        ubound = upper_bound(n._lbound, n._rbound, u)
        heapq.heappush(heap_n, (-1 * ubound, n))

        while (len(heap_n) > 0):
            v_n, n = heapq.heappop(heap_n)
            if (n._lchild is None):  # leaf node
                for tid in n._tids:
                    value = self._Q[tid].dot(u)
                    if (len(heap_r) < k):
                        heapq.heappush(heap_r, (value, tid))
                    else:
                        if (value > heap_r[0][0]):
                            heapq.heappushpop(heap_r, (value, tid))
                if (len(heap_r) >= k and heap_r[0][0] >= -1 * v_n):
                    break
            else:  # None leaf node
                lc_n = n._lchild
                ubound = upper_bound(lc_n._lbound, lc_n._rbound, u)
                heapq.heappush(heap_n, (-1 * ubound, lc_n))
                rc_n = n._rchild
                ubound = upper_bound(rc_n._lbound, lc_n._rbound, u)
                heapq.heappush(heap_n, (-1 * ubound, rc_n))

        res = []
        for r in heap_r:
            res.append(r[1])
        return res

    def print_index(self):
        n_stack = [(self._n_root, 0)]
        while (len(n_stack) > 0):
            n, l = n_stack.pop()
            line = ""
            for _ in xrange(l * 2):
                line += " "
            line += str(n._count) + ", " + str(n._lbound) + ", " + str(n._rbound)
            print line
            if (not n._rchild is None):
                n_stack.append((n._rchild, l + 1))
                n_stack.append((n._lchild, l + 1))


if __name__ == "__main__":
    """
    Unit test procedure
    """
    T = KDTree()
    Q = np.random.random_sample([100, 2])
    # print Q

    T.create_index(Q, 100, 2)
    # T.print_index()

    u = np.array([0.1, -0.1])
    maxval = -1
    for i in xrange(100):
        maxval = max(maxval, Q[i].dot(u))
    print maxval

    res = T.top_item(u, 2)
    print Q[res[1]]
    print Q[res[1]].dot(u)
