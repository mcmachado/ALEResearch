Changes are in four files.
First, is the parameters.cpp/hpp. It will catch one more command option -n, which will turn on the newly-added checkPoint-saving and checkPoint-loading function.
After the -n, you need to provide the name for checkPoints.

Second, is the rlLearner.cpp/hpp. I changed the epsilonGreedy function, so now it does not use rand. Instead, it uses the <random> modules provided by 
c++11.

Third, is the sarsaLearner.cpp/hpp. I simply added checkPoint saving and loading functions in. So any other functions you have implemented are still there.

Forth, is the BPROFeature.cpp/hpp. I almost rewrite the whole feature generation part. So now it does not have any redundant features.

Actually, there are two kinds of checkPoints in my program. The first kind is simply called checkPoint, where random generator state/ total frames/ first reward / weights,etc. are saved. The second kind is called learningCondition, where each episode’s result is saved.
The program will append new episode’s result to the learningCondition file. So there is only one such file for each run. However, the name will be changed to reflect how many episodes have been passed.
For the checkPoint thing, the program will write a new checkPoint when it’s time to save, and it will remove all older versions of checkPoints. So still the program will only keep one copy of checkPoint, and it is always the most recent one.

You still need to provide the main file. And your main file is totally compatible with my changes.

An example for how to use the checkPoint saving/loading function. Let’s say the main file generates an executable file called learnerBpro. So the command will be:
./learnerBpro -s 1 -r ../roms/Asterix.bin -c bpro.cfg -n Bpro-Asterix-Trial1
In this case, the checkPoint file will be named as Bpro-Asterix-Trial1-checkPoint-Episode*-finished.txt (where * indicates how many episodes have been passed, finished means, the file is not currently  being written)
The learningCondition file will be named as Bpro-Asterix-Trial1-learningConditon-Episode*-finished.txt

I also provide my main file, so you can see what has been changed. Basically it is identical with your main file.

If you have any questions, feel free to shoot me an email.

P.S The dependencies in your makefile is not correct, so ever ytime, it will compile the whole thing. I also put the corrected version of makefile in the folder.



