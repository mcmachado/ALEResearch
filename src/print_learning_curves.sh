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
        #echo "set terminal wxt size 350,262 enhanced font 'Verdana,10' persist"  >> ${name}.printscript
        echo "set output 'learning_curve_${name}.pdf'">> ${name}.printscript
        echo 'set ylabel "average reward"'>> ${name}.printscript
        echo 'set xlabel "episode"'>> ${name}.printscript
        echo 'set key font "sans,7"' >> ${name}.printscript
        echo "set style line 101 lc rgb '#808080' lt 1 lw 1" >> ${name}.printscript
        echo "set border 3 front ls 101" >> ${name}.printscript
        echo "set tics nomirror out scale 0.75" >> ${name}.printscript
        echo "set format '%g'" >> ${name}.printscript
        echo "load 'parula.pal'" >> ${name}.printscript
        echo "set palette rgbformulae 7,5,15"  >> ${name}.printscript
        echo "plot '< awk -vn=1000 -f average_all.awk ${file}.tmp' using 1:2 title 'mode ${mode} difficulty ${difficulty}' with lines ls 1" >> ${name}.printscript
    else
        mv ${name}.printscript ttmp.printscript
        escaped=`ls "${file}.tmp"| sed 's#/#\\\/#g'`
        last_num=`cat ttmp.printscript | pcregrep -o1 'lines ls ([0-9]*)$'`
        #echo "s/lines$/lines, '< awk -vn=1000 -f average.awk ${escaped}' using 1:2 title 'mode ${mode} difficulty ${difficulty}' with lines/g" ttmp.printscript
        next_num=$(($last_num+1))
        sed "s/lines ls ${last_num}$/lines ls ${last_num}, '< awk -vn=1000 -f average_all.awk ${escaped}' using 1:2 title 'mode ${mode} difficulty ${difficulty}' with lines ls ${next_num}/g" ttmp.printscript > ${name}.printscript
        rm ttmp.printscript
    fi
    
done

gnuplot -p *.printscript
rm -rf *.printscript
