/****************************************************************************************
** Starting point for running Sarsa algorithm. Here the parameters are set, the algorithm
** is started, as well as the features used. In fact, in order to create a new learning
** algorithm, once its class is implementend, the main file just need to instantiate
** Parameters, the Learner and the type of Features to be used. This file is a good 
** example of how to do it. A parameters file example can be seen in ../conf/sarsa.cfg.
** This is an example for other people to use: Sarsa with Basic Features.
** 
** Author: Marlos C. Machado
***************************************************************************************/

#ifndef ALE_INTERFACE_H
#define ALE_INTERFACE_H
#include <ale_interface.hpp>
#endif
#ifndef PARAMETERS_H
#define PARAMETERS_H
#include "../../src/common/Parameters.hpp"
#endif
#ifndef Q_H
#define Q_H
#include "agents/rl/qlearning/QLearner.hpp"
#endif
#ifndef BPRO_FEATURES_H
#define BPRO_FEATURES_H
#include "../../src/features/BPROFeatures.hpp"
//#include "../../src/features/BasicFeatures.hpp"
#endif

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
	//Using RAM features:
	BPROFeatures features(&param);
	//BasicFeatures features(&param);
	//Reporting parameters read:
	printBasicInfo(param);

	ALEInterface ale(param.getDisplay());
	ale.loadROM(param.getRomPath().c_str());

	//Instantiating the learning algorithm:
	QLearner q(ale, &features, &param);
    //Learn a policy:
    q.learnPolicy(ale, &features);
    printf("\n\n== Evaluation without Learning == \n\n");
    q.evaluatePolicy(ale, &features);


    return 0;
}
