#ifndef MATHEMATICS_H
#define MATHEMATICS_H
#include "../../../../../src/common/Mathematics.hpp"
#endif

#ifndef RL_LEARNER_H
#define RL_LEARNER_H
#include "RLLearnerExtended.hpp"
#endif

#include <fstream>

#define NUM_BITS 1024

RLLearner::RLLearner(ALEInterface& ale, Features *features, Parameters *param){
	randomActionTaken   = 0;

	gamma               = param->getGamma();
	epsilon             = param->getEpsilon();
	toUseOnlyRewardSign = param->getUseRewardSign();
	toBeOptimistic      = param->getOptimisticInitialization();
	
	episodeLength       = param->getEpisodeLength();
	numEpisodesEval     = param->getNumEpisodesEval();
	totalNumberOfFramesToLearn = param->getLearningLength();

	for(int i = 0; i < (ramFeatures.getNumberOfFeatures() - 1)*2; i++){
		transitions.push_back(0);
	}

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

	//Reading file containing the vector that describes the reward for the option learning
	//The first X positions encode the transition 0->1 and the other X encode 1->0.
	pathToRewardDescription = param->getOptionRewardPath();
	std::ifstream infile1(pathToRewardDescription.c_str());
	float value;
	while(infile1 >> value){
		option.push_back(value);
	}
	pathToStatsDescription = param->getDataStatsPath();
	std::ifstream infile2((pathToStatsDescription + "_mean.out").c_str());
	while(infile2 >> value){
		mean.push_back(value);
	}
	std::ifstream infile3((pathToStatsDescription + "_std.out").c_str());
	while(infile3 >> value){
		std.push_back(value);
	}
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

int RLLearner::playOption(ALEInterface& ale, int option, Features *features,
	vector<vector<vector<float> > > &learnedOptions){

	int r_real = 0;
	float termProb = 0.01;
	int currentAction;
	vector<int> Fbpro;	                      //Set of features active
	vector<float> Q(numBasicActions, 0.0);    //Q(a) entries

	RAMFeatures ramFeatures;
	vector<bool> F(NUM_BITS, 0); //Set of active features
	vector<bool> Fprev;

	while(rand()%1000 > 1000 * termProb && !ale.game_over()){
		//Get state and features active on that state:		
		Fbpro.clear();
		features->getActiveFeaturesIndices(ale.getScreen(), ale.getRAM(), Fbpro);

		//Update Q-values for each possible action
		for(int a = 0; a < numBasicActions; a++){
			float sumW = 0;
			for(unsigned int i = 0; i < Fbpro.size(); i++){
				sumW += learnedOptions[option][a][Fbpro[i]];
			}
			Q[a] = sumW;
		}

		currentAction = epsilonGreedy(Q);
		//Take action, observe reward and next state:
		r_real += ale.act((Action) currentAction);
		Fprev.swap(F);
		F.clear();
		ramFeatures.getCompleteFeatureVector(ale.getScreen(), ale.getRAM(), F);
		F.pop_back();
	}
	return r_real;
}

void RLLearner::updateTransitionVector(vector<bool> F, vector<bool> Fnext, vector<float>& transitions){
	int numTransitionFeatures = F.size();
	
	for(int i = 0; i < F.size(); i++){
		if(!F[i] && Fnext[i]){ //0->1
			transitions[i] = 1;
		}
		else{
			transitions[i] = 0;	
		}
		if(F[i] && !Fnext[i]){ //1->0
			transitions[i + numTransitionFeatures - 1] = 1;
		}
		else{
			transitions[i + numTransitionFeatures - 1] = 0;	
		}
	}
}

/**
 * The first parameter is the one that is used by Sarsa. The second is used to
 * pass aditional information to the running algorithm (like 'real score' if one
 * is using a surrogate reward function).
 */
void RLLearner::act(ALEInterface& ale, int action, vector<float>& transitions, Features *features,
	vector<float> &reward, vector<vector<vector<float> > > &learnedOptions){

	float r_alg = 0.0, r_real = 0.0;

	FRam.clear();
	ramFeatures.getCompleteFeatureVector(ale.getScreen(), ale.getRAM(), FRam);

	if(action < numBasicActions){
		r_real = ale.act(actions[action]);
	} 
	else{
		int option_idx = action - numBasicActions;
		r_real = playOption(ale, option_idx, features, learnedOptions);
	}

	FnextRam.clear();
	ramFeatures.getCompleteFeatureVector(ale.getScreen(), ale.getRAM(), FnextRam);
	updateTransitionVector(FRam, FnextRam, transitions);

	for(int i = 0; i < transitions.size(); i++){
		transitions[i] = (transitions[i] - mean[i])/std[i];
		r_alg += option[i] * transitions[i];
	}

	reward[0] = r_alg;
	reward[1] = r_real;
}

