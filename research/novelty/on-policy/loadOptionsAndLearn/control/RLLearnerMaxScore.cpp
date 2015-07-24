#ifndef MATHEMATICS_H
#define MATHEMATICS_H
#include "../../../../../src/common/Mathematics.hpp"
#endif

#ifndef RL_LEARNER_H
#define RL_LEARNER_H
#include "RLLearnerMaxScore.hpp"
#endif

#include <fstream>

using namespace std;

RLLearner::RLLearner(ALEInterface& ale, Features *features, Parameters *param){
	randomActionTaken   = 0;

	gamma               = param->getGamma();
	epsilon             = param->getEpsilon();
	toUseOnlyRewardSign = param->getUseRewardSign();
	toBeOptimistic      = param->getOptimisticInitialization();
	
	episodeLength       = param->getEpisodeLength();
	numEpisodesEval     = param->getNumEpisodesEval();
	totalNumberOfFramesToLearn = param->getLearningLength();

	termProb 			= 0.01; //I am testing it twice, so in fact this is 0.01

	//Get the number of effective actions:
	if(param->isMinimalAction()){
		actions = ale.getMinimalActionSet();
	}
	else{
		actions = ale.getLegalActionSet();
	}
	numOptions      = param->getNumOptionsLoad();
	numBasicActions = actions.size();
	numTotalActions = numBasicActions + numOptions;
}

int RLLearner::epsilonGreedy(vector<float> &QValues){
	randomActionTaken = 0;

	int action = Mathematics::argmax(QValues);
	//With probability epsilon: a <- random action in A(s)
	int random = rand();
	if((random % int(nearbyint(1.0/epsilon))) == 0) {
	//if((rand()%int(1.0/epsilon)) == 0){
		randomActionTaken = 1;
		action = rand() % QValues.size();
	}
	return action;
}

void RLLearner::playOption(ALEInterface& ale, int option, Features *features,
	vector<float> &reward, vector<vector<vector<float> > > &learnedOptions){
	int numActionsInOption = learnedOptions[option].size();
	int r_real = 0;
	int currentAction;
	vector<int> Fbpro;	                      //Set of features active
	vector<float> Q(numActionsInOption, 0.0);    //Q(a) entries

	while(rand()%1000 > 1000 * termProb && !ale.game_over()){
		//Get state and features active on that state:		
		Fbpro.clear();
		features->getActiveFeaturesIndices(ale.getScreen(), ale.getRAM(), Fbpro);

		//Update Q-values for each possible action
		for(int a = 0; a < numActionsInOption; a++){
			float sumW = 0;
			for(unsigned int i = 0; i < Fbpro.size(); i++){
				sumW += learnedOptions[option][a][Fbpro[i]];
			}
			Q[a] = sumW;
		}

		currentAction = epsilonGreedy(Q);
		/* Now things get nasty. We need to do it recursively because one 
		option can call another one. Hopefully everything is going to work.*/
		this->act(ale, currentAction, features, reward, learnedOptions);
	}
}

/**
 * The first parameter is the one that is used by Sarsa. The second is used to
 * pass aditional information to the running algorithm (like 'real score' if one
 * is using a surrogate reward function).
 */
void RLLearner::act(ALEInterface& ale, int action, Features *features,
	vector<float> &reward, vector<vector<vector<float> > > &learnedOptions){

	float r_alg = 0.0, r_real = 0.0;

	if(action < numBasicActions){
		r_real = ale.act(actions[action]);
	} 
	else{
		int option_idx = action - numBasicActions;
		playOption(ale, option_idx, features, reward, learnedOptions);
	}
	/* Here I am letting the option return a single reward and then I am
	 normalizing over it. I don't know if it is not better to normalize
	 each step of the option. */
	if(r_real != 0.0){
		if(!sawFirstReward){
			firstReward = std::abs(r_real);
			sawFirstReward = 1;
		}
	}
	if(sawFirstReward){
		r_alg = r_real/firstReward;
	}

	reward[0] += r_alg;
	reward[1] += r_real;
}

