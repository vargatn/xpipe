#!/bin/bash

python=~/anaconda3/envs/main3/bin/python
for i in 0 1 2
do
#  screen -dm bash -c  "conda activate main3; sleep 10; python run_quintile.py --zbin=$i"
  screen -dm bash -c  "
                      source activate main3;
                      $python run_quintile.py --zbin=$i --fiducial --refs --no_overwrite;
                      $python run_quintile.py --zbin=$i --no_overwrite --exts;
                      $python run_quintile.py --zbin=$i --no_overwrite --effs;
                      "
done