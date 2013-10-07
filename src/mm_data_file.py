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

        for line in fid:
            if line[0] = '\%':
                continue
            line = line.rstrip()
            fields = line.split()
            
            self.ratings.append([int(fields[0]) - 1, int(fields[1]) - 1, float(fields[2])])

        fid.close()
