'''
Prepare movielens data
'''

if __name__ == "__main__":
    data_file = '../data/ml-10M100K/ratings.dat'
    fid = open(data_file, 'r')

    M = 0
    N = 0
    NNZ = 0

    dict_m = {}
    dict_n = {}

    rating_list = []
    
    for line in fid:
        fields = line.split('::')
        if not fields[0] in dict_m:
            M += 1
            dict_m[fields[0]] = 1
        if not fields[1] in dict_n:
            N += 1
            dict_n[fields[1]] = 1
        NNZ += 1
        rating_list.append([fields[0], fields[1], fields[2]])

    fid.write('%%MatrixMarket matrix coordinate real general')
    fid.write(str(M) + ' ' + str(N) + ' ' + str(NNZ) + '\n')
    for rating in rating_list:
        fid.write(rating[0] + ' ' + rating[1] + ' ' + rating[2] + '\n')

    fid.close()
