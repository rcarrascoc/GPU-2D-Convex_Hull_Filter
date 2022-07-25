#!/bin/bash

#run as sh script/run_alg.sh $((2**25)) $((2**25)) $((2**29)) 5 data/gpu_scan_normal.dat prog 0 0 0
    
STARTN=$1; DN=$2; ENDN=$3 SAMPLES=$4; OUTFILE=${5}; BINARY=${6}; ALG=${7}; SHAPE=${8}; PROB=${9};
TMEAN=0
TVAR=0
TSTDEV=0
TSTERR=0
for N in `seq ${STARTN} ${DN} ${ENDN}`;
do
    echo -n "${N}  " >> ${OUTFILE}
    #echo "${BINARY} $DEV $N ${ALG}"
    M=0; S=0; x=0; y=0; z=0; v=0; w1=0; x1=0; y1=0; z1=0; v1=0; x2=0; w2=0; y2=0; z2=0; v2=0; Mz=0;
    for k in `seq 1 ${SAMPLES}`;
    do
        echo  "./${BINARY} ${N} ${ALG} $SHAPE  $PROB"
        value=`./${BINARY} ${N} ${ALG} $SHAPE  $PROB`
        echo "${value}"
        x="$(cut -d' ' -f1 <<< "$value")"
        y="$(cut -d' ' -f2 <<< "$value")"
        z="$(cut -d' ' -f3 <<< "$value")"
        x1=$(echo "scale=10; $x1+$x" | bc)
        z1=$(echo "scale=10; $z1+$z" | bc)
        oldM=$M;
        M=$(echo "scale=10;  $M+($x-$M)/$k"           | bc)
        S=$(echo "scale=10;  $S+($x-$M)*($x-${oldM})" | bc)
    done
    #echo "done"
    MEAN=$M
    VAR=$(echo "scale=10; $S/(${SAMPLES}-1.0)"  | bc)
    STDEV=$(echo "scale=10; sqrt(${VAR})"       | bc)
    STERR=$(echo "scale=10; ${STDEV}/sqrt(${SAMPLES})" | bc)
    TMEAN=${MEAN}
    TVAR=${VAR}
    TSTDEV=${STDEV}
    TSTERR=${STERR}
    x2=$(echo "scale=10; $x1/$SAMPLES" | bc)
    #echo " "
    echo "---> (MEAN, t1 -> (${TMEAN}[ms], ${TMEAN} $TSTDEV $y $z)"
    echo -n "${TMEAN} $TSTDEV $y $z     " >> ${OUTFILE}
    echo " "
    echo " " >> ${OUTFILE}
done
echo " " >> ${OUTFILE}
echo " "

date