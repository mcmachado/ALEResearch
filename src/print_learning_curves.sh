#!/bin/bash

path=$1
for file in `ls $1/out*.out`;
do
    echo $file
    mode=`ls ${file} |  grep -o '_m[0-9][0-9]*[_.]' | grep -o '[0-9]*' `
    echo "mode ${mode}"
    difficulty=`ls ${file}  | grep -o '_d[0-9][0-9]*_' | grep -o '[0-9]*' `
    echo "difficulty ${difficulty}"
    name=`ls ${file}  | pcregrep -o1 'out_(.*)_d[0-9]'  `
    echo "name ${name}"
    cat $file | grep -o "episode.*[1-9][0-9]* fps" > "${file}.tmp"
    if ! test -e "${name}.printscript"
    then
        echo "set term pdfcairo" >> ${name}.printscript
        echo "set output 'learning_curve_${name}.pdf'">> ${name}.printscript
        echo 'set ylabel "average reward"'>> ${name}.printscript
        echo 'set xlabel "episode"'>> ${name}.printscript
        echo "plot '< awk -vn=100 -f average.awk ${file}.tmp' using 1:2 title 'mode ${mode} difficulty ${difficulty}' with lines" >> ${name}.printscript
    else
        mv ${name}.printscript ttmp.printscript
        escaped=`ls "${file}.tmp"| sed 's#/#\\\/#g'`
        echo "s/lines$/lines, '< awk -vn=100 -f average.awk ${escaped}' using 1:2 title 'mode ${mode} difficulty ${difficulty}' with lines/g" ttmp.printscript 
        sed "s/lines$/lines, '< awk -vn=100 -f average.awk ${escaped}' using 1:2 title 'mode ${mode} difficulty ${difficulty}' with lines/g" ttmp.printscript > ${name}.printscript
        rm ttmp.printscript
    fi
    
done

gnuplot -p *.printscript
rm -rf *.printscript
