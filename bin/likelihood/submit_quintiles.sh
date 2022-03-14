for i in 0 1 2
do
  screen -dm bash -c  "conda activate main3; sleep 5; python run_quintile.py --zbin=$i"
done