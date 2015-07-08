#!/bin/bash

base_path=`pwd`

cmd="./learner -s 3 -c conf.cfg -r /home/alcinos/roms/bank_heist.bin"

for diff in `seq 0 3`;
do
    for m in `seq 0 5`
    do
        mode=$((${m} * 4))
        mkdir -p run_d${diff}_m${mode}
        cp conf.cfg run_d${diff}_m${mode}/conf.cfg
        echo "DIFFICULTY_LEVEL = ${diff}" >> run_d${diff}_m${mode}/conf.cfg
        echo "GAME_MODE = ${mode}" >> run_d${diff}_m${mode}/conf.cfg
        cp skeleton.pbs run_d${diff}_m${mode}/run_d${diff}_m${mode}.pbs
        cp learner run_d${diff}_m${mode}
        echo "cd ${base_path}/run_d${diff}_m${mode}" >>  run_d${diff}_m${mode}/run_d${diff}_m${mode}.pbs
        echo "${cmd} &> out_d${diff}_m${mode}.out" >> run_d${diff}_m${mode}/run_d${diff}_m${mode}.pbs
    done
done
