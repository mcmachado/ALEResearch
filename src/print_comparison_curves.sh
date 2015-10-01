#!/bin/bash

results='results results_VTR results_OFF'
mkdir -p graphs

for dir in $results
do
    for file in `ls ${dir}/outfiles/out*.out`
    do
        echo $file
        mode=`ls ${file} |  grep -o '_m[0-9][0-9]*[_.]' | grep -o '[0-9]*' `
        echo "mode ${mode}"
        difficulty=`ls ${file}  | grep -o '_d[0-9][0-9]*_' | grep -o '[0-9]*' `
        echo "difficulty ${difficulty}"
        if [ `ls ${file} | grep -o -i ram | wc -l` = 1 ]
        then
            feat="RAM"
        else
            feat="BASIC"
        fi
        echo $feat
        name=`ls ${file}  | pcregrep -o1 "out_(.*)_${feat}"  `
        echo "name ${name}"
        cat $file | grep -o "episode.*[1-9][0-9]* fps" > "${file}.tmp"
        fname="graphs/${name}_${difficulty}_${mode}.printscript"

        case $dir in
            "results") title="scratch";;
            "results_VTR") title="relearn";;
            "results_OFF") title="off-policy";;
        esac

        if ! test -e "${fname}"
        then
            echo "set term pngcairo" >> ${fname}
            echo "set output 'graphs/comparison_graph_${name}_d${difficulty}_m${mode}.png'">> ${fname}
            nname=`echo ${name} | sed -e "s/_/ /g" `
            echo "set title '${nname}\\ {/*0.5 mode ${mode} difficulty ${difficulty}}'" >> ${fname}
            echo 'set ylabel "average reward"'>> ${fname}
            echo 'set xlabel "episode"'>> ${fname}
            echo 'set key font "sans,7"' >> ${fname}
            echo "set style line 101 lc rgb '#808080' lt 1 lw 1" >> ${fname}
            echo "set border 3 front ls 101" >> ${fname}
            echo "set tics nomirror out scale 0.75" >> ${fname}
            echo "set format '%g'" >> ${fname}
            echo "load 'parula.pal'" >> ${fname}
            echo "set palette rgbformulae 7,5,15"  >> ${fname}
            echo "plot '< awk -vn=1000 -f average_all.awk ${file}.tmp' using 1:2 title '${feat} ${title}' smooth acsplines with lines ls 1" >> ${fname}
        else
            mv ${fname} ttmp.printscript
            escaped=`ls "${file}.tmp"| sed 's#/#\\\/#g'`
            last_num=`cat ttmp.printscript | pcregrep -o1 'lines ls ([0-9]*)$'`
            next_num=$(($last_num+1))
            sed "s/lines ls ${last_num}$/lines ls ${last_num}, '< awk -vn=1000 -f average_all.awk ${escaped}' using 1:2 title '${feat} ${title}' smooth acsplines with lines ls ${next_num}/g" ttmp.printscript > ${fname}
            rm ttmp.printscript
        fi
        
    done
done
gnuplot -p graphs/*.printscript
rm -rf graphs/*.printscript

for dir in $results
do
    for file in `ls ${dir}/outfiles/out*.out`
    do
        echo $file
        mode=`ls ${file} |  grep -o '_m[0-9][0-9]*[_.]' | grep -o '[0-9]*' `
        echo "mode ${mode}"
        difficulty=`ls ${file}  | grep -o '_d[0-9][0-9]*_' | grep -o '[0-9]*' `
        echo "difficulty ${difficulty}"
        if [ `ls ${file} | grep -o -i ram | wc -l` = 1 ]
        then
            feat="RAM"
        else
            feat="BASIC"
        fi
        echo $feat
        name=`ls ${file}  | pcregrep -o1 "out_(.*)_${feat}"  `
        echo "name ${name}"
        cat $file | grep -o "episode.*[1-9][0-9]* fps" | head -n 100 > "${file}.tmp"
        fname="graphs/${name}_${difficulty}_${mode}.printscript"

        case $dir in
            "results") title="scratch";;
            "results_VTR") title="relearn";;
            "results_OFF") title="off-policy";;
        esac

        if ! test -e "${fname}"
        then
            echo "set term pngcairo" >> ${fname}
            echo "set output 'graphs/comparison_graph_${name}_d${difficulty}_m${mode}_beg.png'">> ${fname}
            nname=`echo ${name} | sed -e "s/_/ /g" `
            echo "set title '${nname}\\ {/*0.5 mode ${mode} difficulty ${difficulty}}'" >> ${fname}
            echo 'set ylabel "average reward"'>> ${fname}
            echo 'set xlabel "episode"'>> ${fname}
            echo 'set key font "sans,7"' >> ${fname}
            echo "set style line 101 lc rgb '#808080' lt 1 lw 1" >> ${fname}
            echo "set border 3 front ls 101" >> ${fname}
            echo "set tics nomirror out scale 0.75" >> ${fname}
            echo "set format '%g'" >> ${fname}
            echo "load 'parula.pal'" >> ${fname}
            echo "set palette rgbformulae 7,5,15"  >> ${fname}
            echo "plot '< awk -vn=5 -f average_all.awk ${file}.tmp' using 1:2 title '${feat} ${title}' smooth acsplines with lines ls 1" >> ${fname}
        else
            mv ${fname} ttmp.printscript
            escaped=`ls "${file}.tmp"| sed 's#/#\\\/#g'`
            last_num=`cat ttmp.printscript | pcregrep -o1 'lines ls ([0-9]*)$'`
            next_num=$(($last_num+1))
            sed "s/lines ls ${last_num}$/lines ls ${last_num}, '< awk -vn=5 -f average_all.awk ${escaped}' using 1:2 title '${feat} ${title}' smooth acsplines with lines ls ${next_num}/g" ttmp.printscript > ${fname}
            rm ttmp.printscript
        fi
        
    done
done
gnuplot -p graphs/*.printscript
rm -rf *.printscript
