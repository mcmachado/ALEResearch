/******************************************************************************************
** Starting point for running Sarsa algorithm. In this code I am replacing the real      **
** reward function (the score) by a proxy function, the linear combination of my novelty **
** measure. This is a preliminary test to see what type of policy is learned with such a **
** function. But besides this approach, it also loads a set of weights learned in other  **
** iterations 
**                                                                                       **
** Author: Marlos C. Machado                                                             **
*******************************************************************************************/

#include <ale_interface.hpp>

#ifndef PARAMETERS_H
#define PARAMETERS_H
#include "common/ParametersLoadingWeights.hpp"
#endif
#include "../../../../src/features/BPROFeatures.hpp"
#include "control/OptionSarsaExtended.hpp"

#define NUM_ACTIONS        18

void printBasicInfo(Parameters param){
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

void loadWeights(BPROFeatures *features, Parameters *param, vector<vector<vector<float> > > &w){
	int numFeatures = features->getNumberOfFeatures();

	for(int i = 0; i < param->getNumOptionsLoad(); i++){
		w.push_back(vector< vector<float> >(NUM_ACTIONS, vector<float>(numFeatures, 0.0)));
	}

	for(int i = 0; i < param->getNumOptionsLoad(); i++){
		string line;
		int nActions, nFeatures;
		int j, k;
		float value;

		std::ifstream weightsFile (param->pathToOptionFiles[i].c_str());

		weightsFile >> nActions >> nFeatures;
		assert(nActions == NUM_ACTIONS);
		assert(nFeatures == numFeatures);

		while(weightsFile >> j >> k >> value){
			w[i][j][k] = value;
		}
	}
}

int main(int argc, char** argv){
	vector<vector<vector<float> > > w;  //Theta, weights vector
	//Reading parameters from file defined as input in the run command:
	Parameters param(argc, argv);
	srand(param.getSeed());

	//Using B-PRO features:
	BPROFeatures features(&param);

	//Reporting parameters read:
	printBasicInfo(param);
	loadWeights(&features, &param, w);

	ALEInterface ale(param.getDisplay());

	ale.setFloat("frame_skip", param.getNumStepsPerAction());
	ale.setFloat("stochasticity", 0.00);
	ale.setInt("random_seed", param.getSeed());
	ale.setInt("max_num_frames_per_episode", param.getEpisodeLength());

	ale.loadROM(param.getRomPath().c_str());
	//Instantiating the learning algorithm:
	OptionSarsaExtended optionSarsa(ale, &features, &param);
    //Learn a policy:
    optionSarsa.learnPolicy(ale, &features , w);
    printf("\n\n== Evaluation without Learning == \n\n");
    optionSarsa.evaluatePolicy(ale, &features);
	
    return 0;
}