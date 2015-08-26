/********************************************************************************
 *** This implements Sarsa(lambda) loading a given set of weights representing **
 *** options (added to the action set) and then maximizing the game score.     **
 *** Because we have these options (sometimes composed of other options), we   **
 *** had to deal with each option length. This was done with stochast termin.  **
 *** and also with interrupting options.                                       **
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

	for(int i = param->numOptions - 1; i >= 0; i--){
		string line;
		int nActions, nFeatures;
		int j, k;
		float value;

		std::ifstream weightsFile (param->optionsWgts[i].c_str());

		weightsFile >> nActions >> nFeatures;
		
		/*I cannot make this verification anymore, because we are going to load the first X options (iter. 1)
		  that rely on 18 actions, but then the next Y are going to rely on 18 + X actions, and my parameters
		  do not let me know when this changes. Since I am not going to implement this now, I can only rely
		  in myself when I am passing the parameters. This wouldn't fix it anyway.
		  assert(nActions == NUM_ACTIONS);
		*/
		assert(nFeatures == numFeatures);

		int idx = param->numOptions - 1 - i;
		w.push_back(vector< vector<float> >(nActions, vector<float>(numFeatures, 0.0)));
		while(weightsFile >> j >> k >> value){
			w[idx][j][k] = value;
		}
	}
}

 int main(int argc, char **argv){
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
    learner.evaluatePolicy(ale, w);

	return 0;
 }
 