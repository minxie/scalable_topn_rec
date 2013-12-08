#!/bin/bash

for d in '10' '20' '30' '40' '50'
do 
    python src/run_sgd.py -d $d -m 69878 -n 10677 -l 1e-4 -g 1e-4 -a 0.5 -b 0.01 -c 5 -N 10\
        -tr "data/ml-10M100K/ratings.dat_mm" -te "data/xx.te" -rl "result/movielens_res_static"
done
