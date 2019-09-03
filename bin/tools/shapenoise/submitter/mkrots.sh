# Creates submit sripts for each chunk

NCHUNK=1000
IMIN=0
IMAX=200

for ((i=$IMIN; i<$IMAX; i++))
do
    cat template_rotcalc_job.sh | awk '{gsub(/NCHUNK/, '$NCHUNK'); print}' |  awk '{gsub(/ICHUNK/, '$i'); print}' > rotcalc_job_$i.sh	
    llsubmit rotcalc_job_$i.sh
done
