/****************************************************************************************
** Implementation of Q(lambda). It implements Fig. 8.9 (Linear, gradient-descent 
** Q(lambda)) from the book "R. Sutton and A. Barto; Reinforcement Learning: An 
** Introduction. 1st edition. 1988."
** Some updates are made to make it more efficient, as not iterating over all features.
**
** TODO: Make it as efficient as possible. 
** 
** Author: Marlos C. Machado
***************************************************************************************/

#ifndef MATHEMATICS_H
#define MATHEMATICS_H
#include "../../../../../src/common/Mathematics.hpp"
#endif
#ifndef TIMER_H
#define TIMER_H
#include "../../../../../src/common/Timer.hpp"
#endif
#include "QLearner.hpp"
#include <stdio.h>
#include <math.h>

#define MAX_FRAMES  50000000 
#define MEMORY_SIZE 1000000
#define UPDATE_FREQ 10000
#define BATCH_SIZE  32

unsigned long min (unsigned long a, unsigned long b) {
  return !(b<a)?a:b;
}

QLearner::QLearner(ALEInterface& ale, Features *features, Parameters *param) : RLLearner(ale, param) {
	delta = 0.0;
	idxExperience = 0;
	traceThreshold = param->getTraceThreshold();
	alpha = param->getAlpha();
	lambda = param->getLambda();
	
	numFeatures = features->getNumberOfFeatures();
	
	//Get the number of effective actions:
	if(param->isMinimalAction()){
		actions = ale.getMinimalActionSet();
	}
	else{
		actions = ale.getLegalActionSet();
	}
	numActions = actions.size();
	for(int i = 0; i < numActions; i++){
		//Initialize Q;
		Q.push_back(0);
		Qnext.push_back(0);
		//Initialize e:
		e.push_back(vector<double>(numFeatures, 0.0));
		w.push_back(vector<double>(numFeatures, 0.0));
		w_hat.push_back(vector<double>(numFeatures, 0.0));

		nonZeroElig.push_back(vector<int>());
	}
	//Creating dummy experience, it will later be replaced:
	experience exp;
	exp.action = -1;
	exp.reward = 0.0;
	exp.F = vector<int>();
	exp.Fnext = vector<int>();

	for(int i = 0; i < MEMORY_SIZE; i++){
		memory.push_back(exp);
	}
}

QLearner::~QLearner(){}

void QLearner::updateReplTrace(int action){
	//e <- gamma * lambda * e
	for(unsigned int a = 0; a < nonZeroElig.size(); a++){
		int numNonZero = 0;
	 	for(unsigned int i = 0; i < nonZeroElig[a].size(); i++){
	 		int idx = nonZeroElig[a][i];
	 		//To keep the trace sparse, if it is
	 		//less than a threshold it is zero-ed.
			e[a][idx] = gamma * lambda * e[a][idx];
			if(e[a][idx] < traceThreshold){
				e[a][idx] = 0;
			}
			else{
				nonZeroElig[a][numNonZero] = idx;
		  		numNonZero++;
			}
		}
		nonZeroElig[a].resize(numNonZero);
	}
}

void QLearner::updateQValues(vector<int> &Features, vector<double> &QValues, vector<vector<double> > &w){
	for(int a = 0; a < numActions; a++){
		double sumW = 0;
		for(unsigned int i = 0; i < Features.size(); i++){
			sumW += w[a][Features[i]];
		}
		QValues[a] = sumW;
	}
}

void QLearner::sanityCheck(){
	for(int i = 0; i < numActions; i++){
		if(Q[i] > 10e7 || Q[i] != Q[i] /*NaN*/){
			printf("It seems your algorithm diverged!\n");
			exit(0);
		}
	}
}

void QLearner::storeSample(int currentAction, double reward, vector<int> F, vector<int> Fnext){
	experience exp;
	exp.F = F;
	exp.Fnext = Fnext;
	exp.reward = reward;
	exp.action = currentAction;
	memory[idxExperience%MEMORY_SIZE] = exp;
	idxExperience++;
}

void QLearner::experienceReplay(int maxFeatVectorNorm){
	int nextAction, mSample;
	for(int i = 0; i < BATCH_SIZE; i++){

		mSample = rand() % min((unsigned long) MEMORY_SIZE, idxExperience);
		int a    = memory[mSample].action;

		updateQValues(memory[mSample].F, Q, w);
		
		//If this experience ends in a terminal episode:
		if(memory[mSample].Fnext.size() != 0){ 
			updateQValues(memory[mSample].Fnext, Qnext, w_hat);
		} else{
			for(int j = 0; j < Qnext.size(); j++){
				Qnext[j] = 0;
			}
		}
		nextAction = Mathematics::argmax(Qnext);
		delta = memory[mSample].reward + gamma * Qnext[nextAction] - Q[a];
		
	
		double step_size = alpha/maxFeatVectorNorm;
		for(unsigned int j = 0; j < memory[mSample].F.size(); j++){
			int idx = memory[mSample].F[j];
			w_hat[a][idx] = w_hat[a][idx] + step_size * delta;
		}
	}
}

void QLearner::learnPolicy(ALEInterface& ale, Features *features){
	vector<double> reward;
	double cumReward = 0, prevCumReward = 0;
	sawFirstReward = 0; firstReward = 1.0;

	unsigned int maxFeatVectorNorm = 1;
	unsigned long totalNumFrames = 0;

	struct timeval tvBegin, tvEnd, tvDiff;
	double elapsedTime;

	//Repeat (for each episode):
	int episode = 0;
	while(totalNumFrames < MAX_FRAMES){
		gettimeofday(&tvBegin, NULL);

		episode++;
		frame = 0;
		
		F.clear();
		features->getActiveFeaturesIndices(ale.getScreen(), ale.getRAM(), F);
		//To ensure the learning rate will never increase along the time:
		if (F.size() > maxFeatVectorNorm){
			maxFeatVectorNorm = F.size();
		}

		//Repeat(for each step of episode) until game is over:
		while(totalNumFrames + frame < MAX_FRAMES && !ale.game_over() && ale.lives() == totalNumLives){
			reward.clear();
			reward.push_back(0.0);
			reward.push_back(0.0);

			updateQValues(F, Q, w);
			sanityCheck();

			//Take action, observe reward and next state:
			currentAction = epsilonGreedy(Q);
			act(ale, currentAction, reward);
			cumReward  += reward[1];

			if(!ale.game_over() && ale.lives() == totalNumLives){
				//Obtain active features in the new state:
				Fnext.clear();
				features->getActiveFeaturesIndices(ale.getScreen(), ale.getRAM(), Fnext);
				storeSample(currentAction, reward[0], F, Fnext);

				//To ensure the learning rate will never increase along the time:
				if (Fnext.size() > maxFeatVectorNorm){
					maxFeatVectorNorm = Fnext.size();
				}
			}
			else{
				storeSample(currentAction, reward[0], F, vector<int>());
			}

			experienceReplay(maxFeatVectorNorm);
			
			F = Fnext;
			if((totalNumFrames + frame) % UPDATE_FREQ == 0){
				printf("*");
				w = w_hat;
			}
		}
		ale.reset_game();
		totalNumFrames += frame;
		gettimeofday(&tvEnd, NULL);
		timeval_subtract(&tvDiff, &tvEnd, &tvBegin);
		elapsedTime = double(tvDiff.tv_sec) + double(tvDiff.tv_usec)/1000000.0;
		
		double fps = double(frame)/elapsedTime;
		printf("episode: %d,\t%.0f points,\tavg. return: %.1f,\t%d frames,\t%.0f fps\n", 
			episode, (cumReward-prevCumReward), (double)cumReward/(double) episode, frame, fps);

		prevCumReward = cumReward;
	}
}

void QLearner::evaluatePolicy(ALEInterface& ale, Features *features){
	double reward = 0;
	double cumReward = 0; 
	double prevCumReward = 0;

	//Repeat (for each episode):
	for(int episode = 0; episode < numEpisodesEval; episode++){
		//Repeat(for each step of episode) until game is over:
		for(int step = 0; !ale.game_over() && step < episodeLength; step++){
			//Get state and features active on that state:		
			F.clear();
			features->getActiveFeaturesIndices(ale.getScreen(), ale.getRAM(), F);
			updateQValues(F, Q, w);       //Update Q-values for each possible action
			currentAction = epsilonGreedy(Q);
			//Take action, observe reward and next state:
			reward = 0;
			for(int i = 0; i < numStepsPerAction && !ale.game_over() ; i++){
				reward += ale.act(actions[currentAction]);
			}
			cumReward  += reward;
		}
		ale.reset_game();
		sanityCheck();
		
		printf("%d, %f, %f\n", episode + 1, (double)cumReward/(episode + 1.0), cumReward-prevCumReward);
		
		prevCumReward = cumReward;
	}
}
