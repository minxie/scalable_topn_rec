'''
Prepare movielens data
'''

if __name__ == "__main__":
    data_file = '../data/ml-10M100K/ratings.dat'
    damm_file = '../data/ml-10M100K/ratings.dat_mm'
    fid = open(data_file, 'r')
    oid = open(damm_file, 'w')

    M = 0
    N = 0
    NNZ = 0

    dict_m = {}
    dict_n = {}

    rating_list = []
    
    for line in fid:
        if NNZ % 100000 == 0:
            print str(NNZ)
        fields = line.split('::')
        if not fields[0] in dict_m:
            M += 1
            dict_m[fields[0]] = 1
        if not fields[1] in dict_n:
            N += 1
            dict_n[fields[1]] = 1
        NNZ += 1
        rating_list.append([fields[0], fields[1], fields[2]])

    oid.write('%%MatrixMarket matrix coordinate real general')
    oid.write(str(M) + ' ' + str(N) + ' ' + str(NNZ) + '\n')
    for rating in rating_list:
        oid.write(rating[0] + ' ' + rating[1] + ' ' + rating[2] + '\n')

    fid.close()
    oid.close()
