import numpy as np
cimport numpy as np
from operator import itemgetter

def LowerBound(u, theta, t, D, np.ndarray[np.float64_t, ndim=2] P, np.ndarray[np.float64_t, ndim=2] Q):
    LowerB = 0
    for d in xrange(D):
        LowerB += min((P[u, d] - theta) * Q[t, d], (P[u, d] + theta) * Q[t, d])
    return LowerB

def UpperBound(t_buf, u, theta, t, t_pre, D, np.ndarray[np.float64_t, ndim=2] P, np.ndarray[np.float64_t, ndim=2] Q,
               np.ndarray[np.float64_t, ndim=1] min_val, np.ndarray[np.float64_t, ndim=1] max_val, c_iter, topn, map_id):
    UpperB = 0

    if c_iter == topn: # Initial setting
        B = np.dot(Q[t].T, P[u])
        weight = np.zeros(D)
        value = np.zeros(D)
        lower = np.zeros(D)
        upper = np.zeros(D)
        np.copyto(weight, P[u])
        np.copyto(value, P[u])
        np.copyto(lower, min_val)
        np.copyto(upper, max_val)

        util = []
        for d in xrange(D):
            if weight[d] < 0:
                weight[d] = -1 * weight[d]
                tmp = lower[d]
                lower[d] = -1 * upper[d]
                upper[d] = -1 * tmp
                value[d] = -1 * value[d]
            B -= weight * lower[d]
            util.append([max((value[d] + theta) / weight[d], (value[d] - theta) / weight[d]), weight[d], upper[d]-lower[d]])
        util.sort(key=itemgetter(0), reverse=True)
        t_buf.p_util[map_id] = util
        t_buf.p_B[map_id] = B

        remaining_B = B
        for d in xrange(D):
            if util[d][1] * util[d][2] > remaining_B:
                UpperB += util[d][0] * remaining_B
                break
            else:
                UpperB += util[d][0] * util[d][1] * util[d][2]
                remaining_B -= util[d][1] * util[d][2]
        t_buf.p_UpperB[map_id] = UpperB
    else:
        # Already have utilities calculated
        util = t_buf.p_util[map_id]
        remaining_B = t_buf.p_B[map_id] + (np.dot(Q[t].T, P[u]) - np.dot(Q[t_pre].T, P[u]))
        for d in xrange(D):
            if util[d][1] * util[d][2] > remaining_B:
                UpperB += util[d][0] * remaining_B
                break
            else:
                UpperB += util[d][0] * util[d][1] * util[d][2]
                remaining_B -= util[d][1] * util[d][2]
        t_buf.p_UpperB[map_id] = UpperB

    return UpperB

class TopNBuffer:

    def __init__(self, M, D, topn):
        self.p_Buffer = [0] * 2 * M
        # self.p_LowerB = [0] * 2 * M
        self.p_ref_pt = [0] * 2 * M
        self.p_UpperB = [0] * 2 * M
        self.p_util = [0] * 2 * M
        self.p_B = [0] * 2 * M
        self.p_Creation_Iter = [0] * 2 * M
        for u in xrange(M):
            self.p_Buffer[u] = []
            # self.p_LowerB[u] = []
            self.p_ref_pt[u] = np.random.random_sample((D))
