#ifndef MATHEMATICS_H
#define MATHEMATICS_H
#include "../../../../../src/common/Mathematics.hpp"
#endif

#ifndef RL_LEARNER_H
#define RL_LEARNER_H
#include "RLLearnerExpReplay.hpp"
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
	vector<float> &reward, vector<vector<vector<float> > > &learnedOptions,
	vector<std::vector<float> > &w){
	int numActionsInOption = learnedOptions[option].size();
	int r_real = 0;
	int currentAction;
	vector<int> Fbpro;	                      	 //Set of features active
	vector<float> Q(numActionsInOption, 0.0);    //Q(a) entries

	float termProb = 0.0;

	/* Rationale: Right now we have 'hierarchical' options. Options in the
	   lower level can act over the 18 actions, and then they should take
	   shorter than options that can call other options. Because the frame
	   skip is 5, what we do is the following: the low-level options have
	   a prob. of 0.05 of finishing, which corresponds on expectation to 20
	   steps (x5 ~ 1.6s). High level options have a termination prob. of 0.2
	   because if they call a low-level option they will have a termination
	   probability of 0.2 * 0.05 in fact (100 steps x 5 ~ 8.3s). */
	if(numActionsInOption > actions.size()){
		termProb = 0.2;
	}
	else{
		termProb = 0.05;
	}

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
		this->act(ale, currentAction, features, reward, learnedOptions, w);
	}
}

/**
 * The first parameter is the one that is used by Sarsa. The second is used to
 * pass aditional information to the running algorithm (like 'real score' if one
 * is using a surrogate reward function).
 */
void RLLearner::act(ALEInterface& ale, int action, Features *features,
	vector<float> &reward, vector<vector<vector<float> > > &learnedOptions,
	vector<std::vector<float> > &w){
	float r_alg = 0.0, r_real = 0.0;

	r_real = ale.act(actions[action]);

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

