# Creates submit sripts for each chunk

NCHUNK=1000

FROM=600
TO=1000

for ((i=$FROM; i<$TO; i++))
do
    echo $i
    cat template_covcalc_job.sh | awk '{gsub(/NCHUNK/, '$NCHUNK'); print}' |  awk '{gsub(/ICHUNK/, '$i'); print}' > covcalc_job_$i.sh	
    llsubmit covcalc_job_$i.sh
done
