#!/bin/bash

for d in '10' '20' '30' '40' '50'
do 
    python src/run_sgd.py -d $d -m 1000990 -n 624961 -l 1e-4 -g 1e-4 -ct 1e-1\
        -tr "data/yahoomusic/ratings_kddcup_mm" -te "data/xx.te" -rl "result/yahoomusic_res"
done
