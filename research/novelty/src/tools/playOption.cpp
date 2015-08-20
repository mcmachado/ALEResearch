/*******************************************************************************
*** This implements an the simple replay of a learned option. We load a set   **
*** of weights to play and also the set of weights that may correspond to the **
*** primitive options (if we are playing an option learned in the second, or  **
*** later iteration).                                                         **
*******************************************************************************/

#include <ale_interface.hpp>

#include "control.hpp"
#include "constants.hpp"
#include "Parameters.hpp"
#include "../features/BPROFeatures.hpp"

using namespace std;

void loadPrimitiveOptions(Parameters param, BPROFeatures *bproFeatures, 
	vector<vector<vector<float> > > &primitiveOptions){
	
	int numFeatures = bproFeatures->getNumberOfFeatures();

	for(int i = 0; i < param.numOptions; i++){
		primitiveOptions.push_back(vector< vector<float> >(NUM_ACTIONS, vector<float>(numFeatures, 0.0)));
	}

	for(int i = 0; i < param.numOptions; i++){
		int j, k;
		float value;
		string line;
		int nActions, nFeatures;

		std::ifstream weightsFile (param.optionsWgts[i].c_str());

		weightsFile >> nActions >> nFeatures;
		assert(nActions == NUM_ACTIONS);
		assert(nFeatures == numFeatures);

		while(weightsFile >> j >> k >> value){
			primitiveOptions[i][j][k] = value;
		}
	}
}

void loadOptionToBePlayed(Parameters param, BPROFeatures *bproFeatures, vector<vector<float> > &w){
	
	int i, j;
	float value;
	int nActions, nFeatures;
	int numTotalActions = NUM_ACTIONS + param.numOptions;
	int numFeatures = bproFeatures->getNumberOfFeatures();

	std::ifstream weightsFile (param.pathOptionToPlay.c_str());

	weightsFile >> nActions >> nFeatures;
	assert(nActions == numTotalActions);
	assert(nFeatures == numFeatures);
	while(weightsFile >> i >> j >> value){
		w[i][j] = value;
	}
}

int main(int argc, char** argv){

	Parameters param(argc, argv);
	srand(param.seed);

	BPROFeatures bproFeatures(param.gameName);
	int numTotalActions = NUM_ACTIONS + param.numOptions;
	int numFeatures = bproFeatures.getNumberOfFeatures();

	vector<int> F;
	vector<float> Q(numTotalActions, 0);
	vector<vector<vector<float> > > primitiveOptions;
	vector<vector<float> > optionBeingPlayed(
		numTotalActions, vector<float>(numFeatures, 0.0));;

	loadOptionToBePlayed(param, &bproFeatures, optionBeingPlayed);
	loadPrimitiveOptions(param, &bproFeatures, primitiveOptions);

	ALEInterface ale(1);
	ale.setInt  ("random_seed"               , param.seed);
	ale.setInt  ("max_num_frames_per_episode", 18000     );
	ale.setBool ("color_averaging"           , true      );
	ale.setFloat("frame_skip"                , 1         );
	ale.setFloat("repeat_action_probability" , 0.00      );

	ale.loadROM(param.romPath.c_str());

	ActionVect actions = ale.getLegalActionSet();

	int currentAction, score = 0;

	while(!ale.game_over()){
		//Get state and features active on that state:		
		F.clear();
		bproFeatures.getActiveFeaturesIndices(ale.getScreen(), F);
		updateQValues(F, Q, optionBeingPlayed);
		currentAction = epsilonGreedy(Q, numTotalActions);
		//Take action, observe reward and next state:
		score += takeAction(ale, bproFeatures, currentAction, actions, primitiveOptions);
	}

	printf("Episode ended with a score of %d points\n", score);
	return 0;
}
