import argparse
import operator


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('-infile', nargs='?', dest='infile', default="", type=str,
                        help='Input File location.')
    parser.add_argument('-oufile', nargs='?', dest='oufile', default="", type=str,
                        help='Output File location.')
    parser.add_argument('-p', nargs='?', dest='p', default=0, type=int,
                        help='Position.')
    args = parser.parse_args()

    iid = open(args.infile, 'r')
    h_counter_position = []
    for i in xrange(10):
        h_counter_position.append({})
    h_counter_global = {}
    
    for line in iid:
        line = line.rstrip()
        fields = line.split()
        for i in xrange(10):
            item = int(fields[i])

            if not item in h_counter_global:
                h_counter_global[item] = 1
            else:
                h_counter_global[item] += 1

            if not item in h_counter_position[i]:
                h_counter_position[i][item] = 1
            else:
                h_counter_position[i][item] += 1

    #print(len(h_counter_global))
    #for i in xrange(10):

    # sorted_x = sorted(h_counter_position[i].iteritems(), key=operator.itemgetter(1))
    # for x in sorted_x:
    #     print x

    sorted_x = sorted(h_counter_global.iteritems(), key=operator.itemgetter(1))
    for x in sorted_x:
        print x
    
    # print(h_counter_position[args.p])

    iid.close()
