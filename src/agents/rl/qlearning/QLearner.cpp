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
#include "../../../common/Mathematics.hpp"
#endif
#ifndef TIMER_H
#define TIMER_H
#include "../../../common/Timer.hpp"
#endif
#include "QLearner.hpp"
#include <stdio.h>
#include <math.h>

QLearner::QLearner(ALEInterface& ale, Features *features, Parameters *param) : RLLearner(ale, param) {
	delta = 0.0;
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

		nonZeroElig.push_back(vector<int>());
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

void QLearner::updateQValues(vector<int> &Features, vector<double> &QValues){
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

void QLearner::learnPolicy(ALEInterface& ale, Features *features){
	struct timeval tvBegin, tvEnd, tvDiff;
	vector<double> reward;
	double elapsedTime;
	double cumReward = 0, prevCumReward = 0;
	unsigned int maxFeatVectorNorm = 1;
	sawFirstReward = 0; firstReward = 1.0;

	//Repeat (for each episode):
	int episode, totalNumberFrames = 0;
	//This is going to be interrupted by the ALE code since I set max_num_frames beforehand
	for(episode = 0; totalNumberFrames < totalNumberOfFramesToLearn; episode++){ 
		
		//We have to clean the traces every episode:
		for(unsigned int a = 0; a < nonZeroElig.size(); a++){
			for(unsigned int i = 0; i < nonZeroElig[a].size(); i++){
				int idx = nonZeroElig[a][i];
				e[a][idx] = 0.0;
			}
			nonZeroElig[a].clear();
		}

		F.clear();
		features->getActiveFeaturesIndices(ale.getScreen(), ale.getRAM(), F);
		//To ensure the learning rate will never increase along
		//the time, Marc used such approach in his JAIR paper		
		if (F.size() > maxFeatVectorNorm){
			maxFeatVectorNorm = F.size();
		}
		gettimeofday(&tvBegin, NULL);

		//This also stops when the maximum number of steps per episode is reached
		while(!ale.game_over()){
			reward.clear();
			reward.push_back(0.0);
			reward.push_back(0.0);
			updateQValues(F, Q);
			sanityCheck();

			//Take action, observe reward and next state:
			currentAction = epsilonGreedy(Q);
			act(ale, currentAction, reward);
			cumReward  += reward[1];

			if(!ale.game_over()){
				//Obtain active features in the new state:
				Fnext.clear();
				features->getActiveFeaturesIndices(ale.getScreen(), ale.getRAM(), Fnext);
				//To ensure the learning rate will never increase along
				//the time, Marc used such approach in his JAIR paper
				if (Fnext.size() > maxFeatVectorNorm){
					maxFeatVectorNorm = Fnext.size();
				}
				updateQValues(Fnext, Qnext);     //Update Q-values for the new active features
				nextAction = Mathematics::argmax(Qnext);
			}
			else{
				nextAction = 0;
				for(unsigned int i = 0; i < Qnext.size(); i++){
					Qnext[i] = 0;
				}
			}
			delta = reward[0] + gamma * Qnext[nextAction] - Q[currentAction];
			
			if(randomActionTaken) {
				for(unsigned int a = 0; a < nonZeroElig.size(); a++){
					for(unsigned int i = 0; i < nonZeroElig[a].size(); i++){
						int idx = nonZeroElig[a][i];
						e[a][idx] = 0.0;
					}
					nonZeroElig[a].clear();
				}
			}
			else{
				updateReplTrace(currentAction);
			}
			//For all i in Fa:
			for(unsigned int i = 0; i < F.size(); i++){
				int idx = F[i];
				//If the trace is zero it is not in the vector
				//of non-zeros, thus it needs to be added
				if(e[currentAction][idx] == 0){
			       nonZeroElig[currentAction].push_back(idx);
			    }
				e[currentAction][idx] = 1;
			}

			//Update weights vector:
			for(unsigned int a = 0; a < nonZeroElig.size(); a++){
				for(unsigned int i = 0; i < nonZeroElig[a].size(); i++){
					int idx = nonZeroElig[a][i];
					w[a][idx] = w[a][idx] + (alpha/(maxFeatVectorNorm)) * delta * e[a][idx];
				}
			}
			F = Fnext;
		}
		gettimeofday(&tvEnd, NULL);
		timeval_subtract(&tvDiff, &tvEnd, &tvBegin);
		elapsedTime = double(tvDiff.tv_sec) + double(tvDiff.tv_usec)/1000000.0;
		
		double fps = double(ale.getEpisodeFrameNumber())/elapsedTime;
		printf("episode: %d,\t%.0f points,\tavg. return: %.1f,\t%d frames,\t%.0f fps\n",
			episode + 1, cumReward - prevCumReward, (double)cumReward/(episode + 1.0),
			ale.getEpisodeFrameNumber(), fps);
		totalNumberFrames += ale.getEpisodeFrameNumber();
		prevCumReward = cumReward;
		ale.reset_game();
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
			updateQValues(F, Q);       //Update Q-values for each possible action
			currentAction = epsilonGreedy(Q);
			//Take action, observe reward and next state:
			reward = ale.act(actions[currentAction]);
			cumReward  += reward;
		}
		ale.reset_game();
		sanityCheck();
		
		printf("%d, %f, %f\n", episode + 1, (double)cumReward/(episode + 1.0), cumReward-prevCumReward);
		
		prevCumReward = cumReward;
	}
}
