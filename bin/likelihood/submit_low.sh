for i in 0 1 2
do
  screen -dm bash -c  "conda activate main3; python run_quintile_lowl.py --lbin=$i --effs --no_overwrite;"
done