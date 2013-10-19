#!/bin/bash

for d in '10' '20' '30' '40' '50'
do 
    python src/run_sgd.py -d $d -m 36492 -n 48277 -l 1e-4 -g 1e-4 \
        -tr "data/flixter/ratings_flixter_mm" -te "data/xx.te" -rl "result/flixter_res"
done
