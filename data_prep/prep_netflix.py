'''
Prepare Netflix data
'''

import glob


if __name__ == "__main__":
    data_files = glob.glob('/ubc/cs/home/m/minxie/Storage/Data/Netflix/download/training_set/*.txt')
    damm_file = '../data/netflix/ratings_netflix_mm'
    oid = open(damm_file, 'w')

    M = 0
    NNZ = 0
    dict_m = {}
    rating_list = []
    
    for data_file in data_files:
        fid = open(data_file, 'r')
        itemid = -1
        
        first_line_flag = 1
        for line in fid:
            if first_line_flag == 1:
                line_content = line.rstrip()
                itemid = line_content[:-1]
                print line_content + itemid
                first_line_flag = 0
                continue
            fields = line.split(',')
            if not fields[0] in dict_m:
                M += 1
                dict_m[fields[0]] = M
            NNZ += 1
            rating_list.append([dict_m[fields[0]], itemid, fields[1]])

        oid.write('%%MatrixMarket matrix coordinate real general\n')
        oid.write(str(M) + ' ' + str(len(data_files)) + ' ' + str(NNZ) + '\n')
        for rating in rating_list:
            oid.write(str(rating[0]) + ' ' + str(rating[1]) + ' ' + rating[2] + '\n')

        fid.close()
        
    oid.close()
