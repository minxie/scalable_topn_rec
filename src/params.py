'''
This object defines the configuration objects which will be used
throughout the program
'''

import argparse


class Params:
    def __init__(self):
        self.p_D = -1 # dimensionality of the latent space
        self.p_M = -1 # number of users
        self.p_N = -1 # number of items

        self.p_lambda = -1.0 # lambda for the regularizer
        self.p_gamma = -1.0 # gamma for the gradient descent method

        self.p_train_f_loc = "" # training file location
        self.p_test_f_loc = "" # testing file location

        self.p_a = -1.0 # initial percentage
        self.p_b = -1.0 # delta percentage
        self.p_c = 0 # # of deltas

        self.p_TopN = 0

    def parse_args(self, title):
        parser = argparse.ArgumentParser(description=title)
        parser.add_argument('-d', nargs='?', dest='d', default=-1, type=int,
                            help='Dimensionality of the latent space.')
        parser.add_argument('-m', nargs='?', dest='m', default=-1, type=int,
                            help='Number of users in the matrix.')
        parser.add_argument('-n', nargs='?', dest='n', default=-1, type=int,
                            help='Number of items in the matrix.')
        parser.add_argument('-l', nargs='?', dest='l', default=1e-3, type=float,
                            help='Lambda for the regularizer.')
        parser.add_argument('-g', nargs='?', dest='g', default=1e-3, type=float,
                            help='Gamma for the gradient descent.')
        parser.add_argument('-i', nargs='?', dest='i', default=50, type=int,
                            help='Max # of iterations.')
        parser.add_argument('-sd', nargs='?', dest='sd', default=0.9, type=float,
                            help='SGD step decrement.')
        parser.add_argument('-ct', nargs='?', dest='ct', default=1e-3, type=float,
                            help='Convergence threshold.')
        parser.add_argument('-tr', nargs='?', dest='tr', default="", type=str,
                            help='Training file location.')
        parser.add_argument('-te', nargs='?', dest='te', default="", type=str,
                            help='Testing file location.')
        parser.add_argument('-rl', nargs='?', dest='rl', default="", type=str,
                            help='Result log file location.')
        parser.add_argument('-a', nargs='?', dest='a', default=0.5, type=float,
                            help='Initial percentage.')
        parser.add_argument('-b', nargs='?', dest='b', default=0.01, type=float,
                            help='Delta percentage.')
        parser.add_argument('-c', nargs='?', dest='c', default=5, type=int,
                            help='# of deltas.')
        parser.add_argument('-N', nargs='?', dest='N', default=10, type=int,
                            help='# of items required.')
        parser.add_argument('-t', nargs='?', dest='t', default=0.001, type=float,
                            help='Theta value.')
        args = parser.parse_args()

        self.p_D = args.d
        self.p_M = args.m
        self.p_N = args.n
        self.p_lambda = args.l
        self.p_gamma = args.g
        self.p_max_i = args.i
        self.p_step_dec = args.sd
        self.p_conv_thres = args.ct
        self.p_train_f_loc = args.tr
        self.p_test_f_loc = args.te
        self.p_res_log_f_loc = args.rl
        self.p_a = args.a
        self.p_b = args.b
        self.p_c = args.c
        self.p_TopN = args.N
        self.p_theta = args.t

    def print_params(self):
        print "Dimensionality of the latent space: " + str(self.p_D)
        print "Number of users in the matrix: " + str(self.p_M)
        print "Number of items in the matrix: " + str(self.p_N)
        print "Lambda for the regularizer: " + str(self.p_lambda)
        print "Gamma for the gradient descent: " + str(self.p_gamma)
        print "Maximum iterations: " + str(self.p_max_i)
        print "Step dec: " + str(self.p_step_dec)
        print "Converge threshold: " + str(self.p_conv_thres)
        print "Training file location: " + str(self.p_train_f_loc)
        print "Testing file location: " + str(self.p_test_f_loc)
        print "Result log file location: " + str(self.p_res_log_f_loc)
        print "Inital percentage: " + str(self.p_a)
        print "Delta percentage: " + str(self.p_b)
        print "Num. of deltas: " + str(self.p_c)
        print "Top-N: " + str(self.p_TopN)
        print "Theta: " + str(self.p_theta)
        
