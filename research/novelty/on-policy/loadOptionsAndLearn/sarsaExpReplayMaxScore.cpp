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
#include "../../../../src/features/BPROFeatures_Old.hpp"
#include "control/SarsaLearnerExpReplay.hpp"

#define NUM_ACTIONS        18

using namespace std;

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
	}

	for(int i = param->getNumOptionsLoad()-1; i >= 0; i--){
		string line;
		int nActions, nFeatures;
		int j, k;
		float value;

		std::ifstream weightsFile (param->pathToOptionFiles[i].c_str());

		weightsFile >> nActions >> nFeatures;
		
		/*I cannot make this verification anymore, because we are going to load the first X options (iter. 1)
		  that rely on 18 actions, but then the next Y are going to rely on 18 + X actions, and my parameters
		  do not let me know when this changes. Since I am not going to implement this now, I can only rely
		  in myself when I am passing the parameters. This wouldn't fix it anyway.
		  assert(nActions == NUM_ACTIONS);
		*/
		assert(nFeatures == numFeatures);

		int idx = param->getNumOptionsLoad() - 1 - i;
		w.push_back(vector< vector<float> >(nActions, vector<float>(numFeatures, 0.0)));
		while(weightsFile >> j >> k >> value){
			w[idx][j][k] = value;
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
	ale.setFloat("repeat_action_probability", 0.00);
	ale.setInt("random_seed", param.getSeed());
	ale.setInt("max_num_frames_per_episode", param.getEpisodeLength());

	ale.loadROM(param.getRomPath().c_str());
	//Instantiating the learning algorithm:
	SarsaExpReplay expReplaySarsa(ale, &features, &param);
    //Learn a policy:
    expReplaySarsa.learnPolicy(ale, &features , w);
    //printf("\n\n== Evaluation without Learning == \n\n");
    //expReplaySarsa.evaluatePolicy(ale, &features);
	
    return 0;
}
