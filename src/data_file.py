'''
This file defines the common interfaces for file operation
'''

class DataFile:
    def __init__(self):
        self.ratings = []
        self.iterator = 0

    def reinit(self):
        self.ratings = []
        self.iterator = 0
        
    def read_file(self, params):
        pass

    def next_rating(self):
        if (self.iterator == len(self.ratings)):
            return -1
        return self.ratings[self.iterator]
        self.iterator += 1
