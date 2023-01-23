DATE=$1
SAMPLE=$2
TYPE=$3

for CHR in {1..22} X
do
    python3 plotqc.py $DATE $SAMPLE $TYPE chr$CHR &
done

trap 'kill 0' INT   # make ^C work
status=0            # exit status of this script, assume okay
while true; do
    wait -n                     # wait for any child
    sts=$?                      # capture exit status of wait
    (($sts == 127)) && break    # if 127, no more children
    (($sts)) && status=1        # otherwise exit status of child. if bad, propagate
done
