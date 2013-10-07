'''
This defines the file object to read Matrix Market format
'''

from data_file import DataFile


class MMDataFile(DataFile):
    def __init__(self):
        DataFile.__init__(self)

    def read_file(self, params):
        data_file = params.p_train_f_loc
        fid = open(data_file, 'r')

        line_num = 1
        for line in fid:
            if line_num <= 2:
                line_num += 1
                continue
            line = line.rstrip()
            fields = line.split()

            # Check whether there will be some weird ratings ...
            if (float(fields[2]) > 100):
                print line
                print line_num
                print fields[0]
                print fields[1]
                print fields[2]
                input("Debug:")
            
            self.ratings.append([int(fields[0]) - 1, int(fields[1]) - 1, float(fields[2])])

        fid.close()
