#!/bin/bash

for d in '10' '20' '30' '40' '50'
do 
    python src/run_sgd.py -d 50 -m 46932 -n 100922 -l 1e-4 -g 1e-4 \
        -tr "data/epinion/ratings_epinion_mm" -te "data/xx.te" -rl "result/epinion_res"
done
