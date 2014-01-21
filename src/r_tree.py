import heapq
import math
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
        self._children = []
        self._lbound = None
        self._rbound = None
        self._count = 0
        self._tids = None


class RTree:

    Name = "RTree"

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
        # print lbound
        # print rbound

        max_val = -10000
        max_idx = 0
        for mi in xrange(m):
            if rbound[mi] - lbound[mi] > max_val:
                max_idx = mi

        NODE_SIZE = 50
        tids = range(n)
        copy = [(tid, Q[tid][max_idx]) for tid in tids]
        copy = sorted(copy, key=itemgetter(1))

        lnode_list = []
        for lnodei in xrange(int(math.ceil(n / NODE_SIZE))):
            l_tids = [x for x, y in copy[lnodei * NODE_SIZE:min(lnodei * NODE_SIZE + NODE_SIZE + 1, n)]]
            lbound = np.array([10000] * m)
            rbound = np.array([-10000] * m)
            for tid in l_tids:
                lbound = np.minimum(Q[tid], lbound)
                rbound = np.maximum(Q[tid], rbound)

            node = Node()
            node._lbound = lbound
            node._rbound = rbound
            node._count = len(l_tids)
            node._tids = l_tids
            lnode_list.append(node)

        cnode_list = []
        while len(lnode_list) > NODE_SIZE:
            for cnodei in xrange(int(math.ceil(len(lnode_list) / NODE_SIZE))):
                node = Node()
                lbound = np.array([10000] * m)
                rbound = np.array([-10000] * m)
                for lnodei in xrange(cnodei * NODE_SIZE, min(cnodei * NODE_SIZE + NODE_SIZE + 1, len(lnode_list))):
                    node._children.append(lnode_list[lnodei])
                    lbound = np.minimum(lnode_list[lnodei]._lbound, lbound)
                    rbound = np.maximum(lnode_list[lnodei]._rbound, rbound)
                    node._count += lnode_list[lnodei]._count
                node._lbound = lbound
                node._rbound = rbound
                cnode_list.append(node)
            lnode_list = cnode_list
            cnode_list = []

        n_root = Node()
        lbound = np.array([10000] * m)
        rbound = np.array([-10000] * m)
        for lnode in lnode_list:
            n_root._children.append(lnode)
            lbound = np.minimum(lnode._lbound, lbound)
            rbound = np.maximum(lnode._rbound, rbound)
            n_root._count += lnode._count
        n_root._lbound = lbound
        n_root._rbound = rbound

        self._n_root = n_root

    def top_item(self, u, k):
        heap_n = []  # a priority queue for index nodes
        heap_r = []  # a priority queue for candidate results

        n = self._n_root
        ubound = upper_bound(n._lbound, n._rbound, u)
        heapq.heappush(heap_n, (-1 * ubound, n))

        while (len(heap_n) > 0):
            v_n, n = heapq.heappop(heap_n)
            if not n._children:  # leaf node
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
                for child in n._children:
                    ubound = upper_bound(child._lbound, child._rbound, u)
                    heapq.heappush(heap_n, (-1 * ubound, child))

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
            for child in n._children:
                n_stack.append((child, l + 1))

if __name__ == "__main__":
    """
    Unit test procedure
    """
    T = RTree()
    Q = np.random.random_sample([10000, 2])
    # print Q

    T.create_index(Q, 10000, 2)
    # T.print_index()

    u = np.array([0.1, -0.1])
    maxval = -1
    for i in xrange(10000):
        maxval = max(maxval, Q[i].dot(u))
    print maxval

    res = T.top_item(u, 1)
    #print Q[res[0]]
    print Q[res[0]].dot(u)
