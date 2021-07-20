#@ job_type = serial
#@ class = serial
#@ group = pr92qe
#
###                   hh:mm:ss
#@ wall_clock_limit = 24:00:00
#
### Don't reserve a node unless you have to.
### One reason not to share could be that you want the full 64 GB of RAM.
#@ node_usage = shared
#@ resources = ConsumableCpus(1) ConsumableMemory(5 gb)
#
#@ job_name = eos-$(jobid)
#@ initialdir = $(home)/DES_Y1A1_cluster/deswlwg-y1redmapper/pipeline/bin/tools/shapenoise/
#
### Want to keep logs in the project directory. Try `echo $WORK` to see your directory
### Use $jobid to have separate logs for each run.
#@ output = /gpfs/work/pr92qe/di49quq/log/$(jobid).out
#@ error  = /gpfs/work/pr92qe/di49quq/log/$(jobid).err
#@ notification = never
#@ notify_user = vargatn@usm.uni-muenchen.de
#@ queue

echo NCHUNK ICHUNK
py=/home/hpc/pr92qe/di49quq/anaconda3/envs/py27/bin/python2

$py shapecov.py --nchunk=NCHUNK --ichunk=ICHUNK

