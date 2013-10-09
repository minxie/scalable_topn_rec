'''
Prepare Netflix data
'''

import glob


if __name__ == "__main__":
    data_files = glob.glob('/ubc/cs/home/m/minxie/Storage/Data/epinion/epinions/epinions/reviews-orig/*')
    damm_file = '../data/epinion/ratings_epinion_mm'
    oid = open(damm_file, 'w')

    M = 0
    N = 0
    NNZ = 0
    dict_n = {}
    rating_list = []

    user_id = 0
    item_id = 0
    for data_file in data_files:
        fid = open(data_file, 'r')

        valid_user_flag = 1
        for line in fid:
            if line == "":
                continue
            fields = line.split('\t')
            if fields[7] == "na":
                continue

            if valid_user_flag == 1:
                user_id += 1
                valid_user_flag = 0

            if not fields[3] in dict_n:
                N += 1
                dict_n[fields[3]] = N
                item_id = N
            else:
                item_id = dict_n[fields[3]]
            NNZ += 1
            rating_list.append([user_id, item_id, fields[7]])
            # oid.write(data_file + '\n')
            # oid.write(str(user_id) + ' ' + fields[3] + ' ' + fields[7] + '\n')

        fid.close()

    oid.write('%%MatrixMarket matrix coordinate real general\n')
    oid.write(str(user_id) + ' ' + str(N) + ' ' + str(NNZ) + '\n')
    for rating in rating_list:
        oid.write(str(rating[0]) + ' ' + str(rating[1]) + ' ' + rating[2] + '\n')
        
    oid.close()
