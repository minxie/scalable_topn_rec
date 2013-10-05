'''
Prepare flixter data
'''

if __name__ == "__main__":
    data_file = '/ubc/cs/home/m/minxie/Storage/Research/ppcf/data/Ratings.timed_processed.txt'
    damm_file = '../data/flixter/ratings_flixter_mm'
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
        fields = line.split('\t')
        if not fields[0] in dict_m:
            M += 1
            dict_m[fields[0]] = M
        if not fields[1] in dict_n:
            N += 1
            dict_n[fields[1]] = N
        NNZ += 1
        rating_list.append([dict_m[fields[0]], dict_n[fields[1]], fields[2]])

    oid.write('%%MatrixMarket matrix coordinate real general\n')
    oid.write(str(M) + ' ' + str(N) + ' ' + str(NNZ) + '\n')
    for rating in rating_list:
        oid.write(str(rating[0]) + ' ' + str(rating[1]) + ' ' + rating[2] + '\n')

    fid.close()
    oid.close()
