import argparse

def read_file(filename, ratings):
    fid = open(filename, 'r')

    for line in fid:
        line = line.rstrip()
        fields = line.split()

        rating = []
        for field in fields:
            rating.append(int(field))
        ratings.append(rating)

    fid.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('-infile', nargs='?', dest='infile', default="", type=str,
                        help='Input File location.')
    parser.add_argument('-oufile', nargs='?', dest='oufile', default="", type=str,
                        help='Output File location.')
    args = parser.parse_args()
    filename = args.infile

    oid = open(args.oufile, 'a')

    topn_res = [[],[],[],[],[],[]]
    read_file(filename+str(0), topn_res[0])
    for iteration in xrange(4):
        read_file(filename+str(iteration+1), topn_res[iteration+1])

        total = [0] * 10
        total_change = [0] * 10
        for u in xrange(len(topn_res[iteration])):
            for pos in xrange(10):
                if topn_res[iteration][pos] != topn_res[iteration+1][pos]:
                    total_change[pos] += 1
                total[pos] += 1

        for pos in xrange(10):
            oid.write(str(total_change[pos]) + '/' + str(total[pos]) + ' ')
        oid.write('\n')

    oid.close()
    
