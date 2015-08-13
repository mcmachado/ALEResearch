#include <random>

#include "control.hpp"
#include "../common/Mathematics.hpp"

using namespace std;

int getNextAction(ALEInterface& ale, int numOptions){
	int totalNumActions = NUM_ACTIONS + numOptions;
	return rand()%totalNumActions;
}

int epsilonGreedy(vector<float> &QValues){
	float epsilon = 0.05;
	int randomActionTaken = 0;

	int action = Mathematics::argmax(QValues);
	//With probability epsilon: a <- random action in A(s)
	int random = rand();
	if((random % int(nearbyint(1.0/epsilon))) == 0) {
		randomActionTaken = 1;
		action = rand() % NUM_ACTIONS;
	}
	return action;
}

void updateQValues(vector<int> &Features, vector<float> &QValues, 
	vector<vector<vector<float> > > &w, int option){

	for(int a = 0; a < NUM_ACTIONS; a++){
		float sumW = 0;
		for(unsigned int i = 0; i < Features.size(); i++){
			sumW += w[option][a][Features[i]];
		}
		QValues[a] = sumW;
	}
}
