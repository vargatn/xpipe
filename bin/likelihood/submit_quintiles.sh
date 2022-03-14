#!/bin/bash
for i in 0 1 2
do
#  screen -dm bash -c  "conda activate main3; sleep 10; python run_quintile.py --zbin=$i"
  screen -dm bash -c "source activate main3; sleep 10; python run_quintile.py --zbin=$i"
done