/********************************************************************************
 *** This implements Sarsa(lambda) loading a given set of weights representing **
 *** options (added to the action set) and then maximizing the 'intrinsic      **
 *** reward' defined as the dot product between the eigenvector passed as      **
 *** parameter and the obtained flip vector. The new option is saved to a file **
 ***                                                                           **
 *** Author: Marlos C. Machado                                                 **
 ********************************************************************************/

#include <vector>

#include "Learner.hpp"
#include "Parameters.hpp"
#include "constants.hpp"
#include "../features/BPROFeatures.hpp"

using namespace std;

void loadWeights(BPROFeatures *features, Parameters *param, vector<vector<vector<float> > > &w){
	int numFeatures = features->getNumberOfFeatures();

	for(int i = 0; i < param->numOptions; i++){
		w.push_back(vector< vector<float> >(NUM_ACTIONS, vector<float>(numFeatures, 0.0)));
	}

	for(int i = 0; i < param->numOptions; i++){
		string line;
		int nActions, nFeatures;
		int j, k;
		float value;

		std::ifstream weightsFile (param->optionsWgts[i].c_str());

		weightsFile >> nActions >> nFeatures;
		assert(nActions == NUM_ACTIONS);
		assert(nFeatures == numFeatures);

		while(weightsFile >> j >> k >> value){
			w[i][j][k] = value;
		}
	}
}

int main(int argc, char** argv){
	//Reading parameters from file defined as input in the run command:
	Parameters param(argc, argv);
	srand(param.seed);

	//Using B-PRO features:
	BPROFeatures bproFeatures(param.gameName);

	vector<vector<vector<float> > > w;
	loadWeights(&bproFeatures, &param, w);

	ALEInterface ale(0);
	ale.setInt  ("random_seed"               , param.seed);
	ale.setInt  ("max_num_frames_per_episode", 18000     );
	ale.setBool ("color_averaging"           , true      );
	ale.setFloat("frame_skip"                , FRAME_SKIP);
	ale.setFloat("repeat_action_probability" , 0.00      );

	ale.loadROM(param.romPath.c_str());

	//Instantiating the learning algorithm:
	Learner learner(ale, &param);
    //Learn a policy:
	learner.learnPolicy(ale, w);
    //Evaluate the policy:
	
    return 0;
}
