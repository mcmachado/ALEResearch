/* Author: Marlos C. Machado */

#include "Agent.hpp"

#define NUM_BITS 1024

Agent::Agent(ALEInterface& ale, Parameters *param) : bproFeatures(param) {
	//Get the number of effective actions:
	if(param->isMinimalAction){
		actions = ale.getMinimalActionSet();
	}
	else{
		actions = ale.getLegalActionSet();
	}

	numberOfOptions          = 0;
	numberOfPrimitiveActions = actions.size();
	numberOfAvailActions     = numberOfPrimitiveActions + numberOfOptions;

	for(int i = 0; i < 2 * NUM_BITS; i++){
		freqOfBitFlips.push_back(0.0);
	}

	for(int i = 0; i < numberOfOptions; i++){
		w.push_back(vector< vector<float> >(numberOfPrimitiveActions, vector<float>(bproFeatures.getNumberOfFeatures(), 0.0)));
	}
}
