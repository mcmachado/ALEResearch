/****************************************************************************************
** Implementation of Sarsa(lambda). It implements Fig. 8.8 (Linear, gradient-descent 
** Sarsa(lambda)) from the book "R. Sutton and A. Barto; Reinforcement Learning: An 
** Introduction. 1st edition. 1988."
** Some updates are made to make it more efficient, as not iterating over all features.
**
** TODO: Make it as efficient as possible. 
** 
** Author: Marlos C. Machado
***************************************************************************************/

#ifndef TIMER_H
#define TIMER_H
#include "../../../../../src/common/Timer.hpp"
#endif
#include "TrueOnlineSarsaLearnerSlow.hpp"
#include <stdio.h>
#include <math.h>

TrueOnlineSarsaLearner::TrueOnlineSarsaLearner(ALEInterface& ale, Features *features, Parameters *param) : RLLearner(ale, param) {
	delta = 0.0;
	
	alpha = param->getAlpha();
	lambda = param->getLambda();
	traceThreshold = param->getTraceThreshold();
	numFeatures = features->getNumberOfFeatures();

	for(int i = 0; i < numActions; i++){
		//Initialize Q;
		Q.push_back(0);
		Qnext.push_back(0);
		//Initialize e:
		e.push_back(vector<double>(numFeatures, 0.0));
		w.push_back(vector<double>(numFeatures, 0.0));

		phi.push_back(vector<int>(numFeatures, 0));
	}

	std::stringstream ss;
	ss << "weights_" << param->getSeed() << ".wgt";
	nameWeightsFile =  ss.str();
}

TrueOnlineSarsaLearner::~TrueOnlineSarsaLearner(){}

void TrueOnlineSarsaLearner::updateQValues(vector<int> &Features, vector<double> &QValues){
	for(int a = 0; a < numActions; a++){
		double sumW = 0;
		for(unsigned int i = 0; i < Features.size(); i++){
			sumW += w[a][Features[i]];
		}
		QValues[a] = sumW;
	}
}

void TrueOnlineSarsaLearner::sanityCheck(){
	for(int i = 0; i < numActions; i++){
		if(fabs(Q[i]) > 10e7 || Q[i] != Q[i] /*NaN*/){
			printf("It seems your algorithm diverged!\n");
			exit(0);
		}
	}
}

void TrueOnlineSarsaLearner::learnPolicy(ALEInterface& ale, Features *features){
	
	struct timeval tvBegin, tvEnd, tvDiff;
	vector<double> reward;
	double elapsedTime;
	double norm_a;
	double q_old, delta_q;
	double cumReward = 0, prevCumReward = 0;
	unsigned int maxFeatVectorNorm = 1;
	sawFirstReward = 0; firstReward = 1.0;

	//Repeat (for each episode):
	for(int episode = 0; episode < numEpisodesLearn; episode++){
		//We have to clean the traces every episode:
		for(unsigned int i = 0; i < e.size(); i++){
			for(unsigned int j = 0; j < e[i].size(); j++){
				e[i][j] = 0.0;
				phi[i][j] = 0;
			}
		}
		F.clear();
		features->getActiveFeaturesIndices(ale.getScreen(), ale.getRAM(), F);
		updateQValues(F, Q);
		currentAction = epsilonGreedy(Q);
		
		q_old = Q[currentAction];

		//Repeat(for each step of episode) until game is over:
		gettimeofday(&tvBegin, NULL);
		frame = 0;
		while(frame < episodeLength && !ale.game_over()){
			reward.clear();
			reward.push_back(0.0);
			reward.push_back(0.0);
			updateQValues(F, Q);
			sanityCheck();
			for(unsigned int i = 0; i < e.size(); i++){
				for(unsigned int j = 0; j < e[i].size(); j++){
					phi[i][j] = 0;
				}
			}
			for(unsigned int i = 0; i < F.size(); i++){	
				phi[currentAction][F[i]] = 1;
			}
			//Take action, observe reward and next state:
			act(ale, currentAction, reward);
			cumReward  += reward[1];
			if(!ale.game_over()){
				//Obtain active features in the new state:
				Fnext.clear();
				features->getActiveFeaturesIndices(ale.getScreen(), ale.getRAM(), Fnext);
				updateQValues(Fnext, Qnext);     //Update Q-values for the new active features
				nextAction = epsilonGreedy(Qnext);
			}
			else{
				nextAction = 0;
				for(unsigned int i = 0; i < Qnext.size(); i++){
					Qnext[i] = 0;
				}
			}
			//To ensure the learning rate will never increase along
			//the time, Marc used such approach in his JAIR paper		
			if (F.size() > maxFeatVectorNorm){
				maxFeatVectorNorm = F.size();
			}

			norm_a = alpha/maxFeatVectorNorm;
			delta_q =  Q[currentAction] - q_old;
			q_old   = Qnext[nextAction];
			delta   = reward[0] + gamma * Qnext[nextAction] - Q[currentAction];

			double dot_e_phi = 0;
			for(unsigned int i = 0; i < F.size(); i++){
				int idx = F[i];
				dot_e_phi += e[currentAction][idx];
			}
			for(unsigned int i = 0; i < e.size(); i++){
				for(unsigned int j = 0; j < e[i].size(); j++){
					e[i][j] = e[i][j] + (1 - norm_a * dot_e_phi) * phi[i][j];
					if(e[i][j] < traceThreshold){
						e[i][j] = 0;
					}
					w[i][j] = w[i][j] + norm_a * delta * e[i][j] + norm_a * delta_q * (e[i][j] - phi[i][j]);
					e[i][j] = gamma * lambda * e[i][j];
					if(e[i][j] < traceThreshold){
						e[i][j] = 0;
					}
				}
			}

			F = Fnext;
			currentAction = nextAction;
		}
		ale.reset_game();
		gettimeofday(&tvEnd, NULL);
		timeval_subtract(&tvDiff, &tvEnd, &tvBegin);
		elapsedTime = double(tvDiff.tv_sec) + double(tvDiff.tv_usec)/1000000.0;
		
		double fps = double(frame)/elapsedTime;
		printf("episode: %d,\t%.0f points,\tavg. return: %.1f,\t%d frames,\t%.0f fps\n", 
			episode + 1, (cumReward-prevCumReward), (double)cumReward/(episode + 1.0), frame, fps);
		prevCumReward = cumReward;
	}
}

void TrueOnlineSarsaLearner::evaluatePolicy(ALEInterface& ale, Features *features){
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
			reward = 0;
			for(int i = 0; i < numStepsPerAction && !ale.game_over() ; i++){
				reward += ale.act(actions[currentAction]);
			}
			cumReward  += reward;
		}
		ale.reset_game();
		sanityCheck();
		
		printf("%d, %f, %f \n", episode + 1, (double)cumReward/(episode + 1.0), cumReward-prevCumReward);
		
		prevCumReward = cumReward;
	}
}
