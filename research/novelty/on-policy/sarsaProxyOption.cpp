/******************************************************************************************
** Starting point for running Sarsa algorithm. In this code I am replacing the real      **
** reward function (the score) by a proxy function, the linear combination of my novelty **
** measure. This is a preliminary test to see what type of policy is learned with such a **
** function. Then later we can try to solve the divergence issue if we decide it is a    **
** good idea and then we go to off-policy learning.                                      **
**                                                                                       **
** Author: Marlos C. Machado                                                             **
*******************************************************************************************/

#include <ale_interface.hpp>

#ifndef PARAMETERS_H
#define PARAMETERS_H
#include "../../../src/common/Parameters.hpp"
#endif
#include "../../../src/features/BPROFeatures.hpp"
#include "control/OptionSarsa.hpp"

void printBasicInfo(Parameters param){
	printf("Seed: %d\n", param.getSeed());
	printf("\nCommand Line Arguments:\nPath to Config. File: %s\nPath to ROM File: %s\nPath to Backg. File: %s\n", 
		param.getConfigPath().c_str(), param.getRomPath().c_str(), param.getPathToBackground().c_str());
	if(param.getSubtractBackground()){
		printf("\nBackground will be subtracted...\n");
	}
	printf("\nParameters read from Configuration File:\n");
	printf("alpha:   %f\ngamma:   %f\nepsilon: %f\nlambda:  %f\nep. length: %d\n\n", 
		param.getAlpha(), param.getGamma(), param.getEpsilon(), param.getLambda(), 
		param.getEpisodeLength());
}

int main(int argc, char** argv){
	//Reading parameters from file defined as input in the run command:
	Parameters param(argc, argv);
	srand(param.getSeed());

	//Using B-PRO features:
	BPROFeatures features(&param);
	//Reporting parameters read:
	printBasicInfo(param);

	ALEInterface ale(param.getDisplay());

	ale.setFloat("frame_skip", param.getNumStepsPerAction());
	ale.setFloat("stochasticity", 0.00);
	ale.setInt("random_seed", param.getSeed());
	ale.setInt("max_num_frames_per_episode", param.getEpisodeLength());

	ale.loadROM(param.getRomPath().c_str());

	//Instantiating the learning algorithm:
	OptionSarsa optionSarsa(ale, &features, &param);
    //Learn a policy:
    optionSarsa.learnPolicy(ale, &features);
    printf("\n\n== Evaluation without Learning == \n\n");
    optionSarsa.evaluatePolicy(ale, &features);
	
    return 0;
}