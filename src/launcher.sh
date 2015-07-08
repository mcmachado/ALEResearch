#!/bin/bash

base_path=`pwd`
game="freeway"
#romPath="/home/alcinos/roms/"
romPath="/home/nicolas/Documents/Etudes/ENS/M1/Stage/emulateur/roms/"
fullPath="${romPath}${game}.bin"
cmd="./learner -s 3 -c conf.cfg -r ${fullPath}"
echo "./mode_difficulty_getter ${fullPath} 1"
for diff in `./mode_difficulty_getter ${fullPath} 1`;
do
    for mode in `./mode_difficulty_getter ${fullPath} 0`;
    do
        folder="${game}_d${diff}_m${mode}"
        mkdir -p ${folder}
        cp conf.cfg ${folder}/conf.cfg
        echo "DIFFICULTY_LEVEL = ${diff}" >> ${folder}/conf.cfg
        echo "GAME_MODE = ${mode}" >> ${folder}/conf.cfg
        cp skeleton.pbs ${folder}/run_d${diff}_m${mode}.pbs
        cp learner ${folder}
        echo "cd ${base_path}/${folder}" >>  ${folder}/run_d${diff}_m${mode}.pbs
        echo "${cmd} &> out_${game}_d${diff}_m${mode}.out" >> ${folder}/run_d${diff}_m${mode}.pbs
        bqsub ${folder}/run_d${diff}_m${mode}.pbs
    done
done


bqsub --submit
