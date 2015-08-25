#include "control.hpp"
#include "constants.hpp"
#include "../common/Mathematics.hpp"

int takeAction(ALEInterface& ale, BPROFeatures features, int actionToTake,
	ActionVect actions, vector<vector<vector<float> > > &primitiveOptions){

	int accumulatedScore = 0;
	//If the selected action was one of the primitive actions
	if(actionToTake < NUM_ACTIONS){
		accumulatedScore += ale.act(actions[actionToTake]);
	}
	else{
		int option = actionToTake - NUM_ACTIONS;
		accumulatedScore = playOption(ale, features, option, actions, primitiveOptions);
	}
	return accumulatedScore;
}

int epsilonGreedy(vector<float> &QValues, int numTotalActions){
	float epsilon = 0.05;
	int randomActionTaken = 0;

	int action = Mathematics::argmax(QValues);
	//With probability epsilon: a <- random action in A(s)
	if(rand()%1000 < 1000 * epsilon) {
		randomActionTaken = 1;
		action = rand() % numTotalActions;
	}
	return action;
}

void updateQValues(vector<int> &Features, vector<float> &QValues, vector<vector<float> > &w){
	//TODO: This will probably break, because I need numTotalActions here
	for(int a = 0; a < NUM_ACTIONS; a++){
		float sumW = 0;
		for(unsigned int i = 0; i < Features.size(); i++){
			sumW += w[a][Features[i]];
		}
		QValues[a] = sumW;
	}
}

int playOption(ALEInterface& ale, BPROFeatures features, int option, 
	ActionVect actions, vector<vector<vector<float> > > &primitiveOptions){

	float epsilon = 0.05;
	int currentAction, cumScore = 0;

	vector<int> F;	//Set of features active
	vector<float> QOptions(NUM_ACTIONS, 0.0);    //Q(a) entries

	while(rand()%1000 > 1000 * PROB_TERMINATION && !ale.game_over()){
		F.clear();
		features.getActiveFeaturesIndices(ale.getScreen(), F);

		updateQValues(F, QOptions, primitiveOptions[option]);
		currentAction = epsilonGreedy(QOptions, NUM_ACTIONS);

		//Take action, observe reward and next state:
		cumScore += ale.act(actions[currentAction]);
	}
	return cumScore;
}
