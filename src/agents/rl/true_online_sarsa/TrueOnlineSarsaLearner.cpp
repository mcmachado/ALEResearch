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

#include "../../../common/Timer.hpp"
#include "TrueOnlineSarsaLearner.hpp"
#include <stdio.h>
#include <math.h>

TrueOnlineSarsaLearner::TrueOnlineSarsaLearner(Environment<bool>& env, Parameters *param, int seed) : RLLearner<bool>(env, param, seed) {
	delta = 0.0;
	
	alpha = param->getAlpha();
	lambda = param->getLambda();
	traceThreshold = param->getTraceThreshold();
	numFeatures = env.getNumberOfFeatures();

	for(int i = 0; i < this->numActions; i++){
		//Initialize Q;
		Q.push_back(0);
		Qnext.push_back(0);
		//Initialize e:
		e.push_back(std::vector<float>(numFeatures, 0.0));
		w.push_back(std::vector<float>(numFeatures, 0.0));

		nonZeroElig.push_back(std::vector<int>());
	}

	std::stringstream ss;
	ss << "weights_" << param->getSeed() << ".wgt";
	nameWeightsFile =  ss.str();
}

TrueOnlineSarsaLearner::~TrueOnlineSarsaLearner(){}

void TrueOnlineSarsaLearner::updateWeights(int action, float alpha, float delta_q){
	for(unsigned int a = 0; a < nonZeroElig.size(); a++){
		for(unsigned int i = 0; i < nonZeroElig[a].size(); i++){
			int idx = nonZeroElig[a][i];
			w[a][idx] = w[a][idx] + alpha * (delta + delta_q) * e[a][idx];
		}
	}

	for(unsigned int i = 0; i < F.size(); i++){
		int idx = F[i];
		w[action][idx] = w[action][idx] - alpha * delta_q;
	}
}

void TrueOnlineSarsaLearner::updateTrace(int action, float alpha){
	float dot_e_phi = 0;
	for(unsigned int i = 0; i < F.size(); i++){
		int idx = F[i];
		dot_e_phi += e[action][idx];
	}

	if((1 - alpha * dot_e_phi) > traceThreshold){
		for(unsigned int i = 0; i < F.size(); i++){
			int idx = F[i];
			if(e[action][idx] == 0){
				nonZeroElig[action].push_back(idx);
			}
			e[action][idx] = e[action][idx] + (1 - alpha * dot_e_phi);
		}
	}
}

void TrueOnlineSarsaLearner::decayTrace(){
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

void TrueOnlineSarsaLearner::sanityCheck(){
	for(int i = 0; i < this->numActions; i++){
		if(fabs(Q[i]) > 10e7 || Q[i] != Q[i] /*NaN*/){
			printf("It seems your algorithm diverged!\n");
			exit(0);
		}
	}
}

void TrueOnlineSarsaLearner::dumpWeights(){
	std::ofstream weightsFile (nameWeightsFile.c_str());
	if(weightsFile.is_open()){
		weightsFile << w.size() << "," << w[0].size() << std::endl;
		for(unsigned int i = 0; i < w.size(); i++){
			for(unsigned int j = 0; j < w[i].size(); j++){
				weightsFile << w[i][j] << std::endl;

			}
		}
		weightsFile.close();
	}
	else{
		printf("Unable to open file to write weights.\n");
	}
}

void TrueOnlineSarsaLearner::loadWeights(){
	std::string line;
	std::ifstream weightsFile (nameWeightsFile.c_str());
	if(weightsFile.is_open()){
		//TODO!!!!
	}
	else{
		printf("Unable to open file to load weights.\n");
	}
}


void TrueOnlineSarsaLearner::learnPolicy(Environment<bool>& env){

	struct timeval tvBegin, tvEnd, tvDiff;
	std::vector<float> reward;
	float elapsedTime;
	float norm_a;
	float q_old, delta_q;
	float cumReward = 0, prevCumReward = 0;
	unsigned int maxFeatVectorNorm = 1;
	this->sawFirstReward = 0; this->firstReward = 1.0;

	//Repeat (for each episode):
	int episode, totalNumberFrames = 0;
	//This is going to be interrupted by the ALE code since I set max_num_frames beforehand
	for(episode = 0; totalNumberFrames < this->totalNumberOfFramesToLearn; episode++){ 
		for(unsigned int a = 0; a < nonZeroElig.size(); a++){
			for(unsigned int i = 0; i < nonZeroElig[a].size(); i++){
				int idx = nonZeroElig[a][i];
				e[a][idx] = 0.0;
			}
			nonZeroElig[a].clear();
		}
		//We have to clean the traces every episode:
		for(unsigned int i = 0; i < e.size(); i++){
			for(unsigned int j = 0; j < e[i].size(); j++){
				e[i][j] = 0.0;
			}
		}
		F.clear();
		env.getActiveFeaturesIndices(F);
		updateQValues(F, w, Q);
		currentAction = epsilonGreedy(Q);
		
		q_old = Q[currentAction];

		//Repeat(for each step of episode) until game is over:
		gettimeofday(&tvBegin, NULL);
		//This also stops when the maximum number of steps per episode is reached
		while(!env.game_over()){
			reward.clear();
			reward.push_back(0.0);
			reward.push_back(0.0);
			updateQValues(F, w, Q);
			sanityCheck();

			//Take action, observe reward and next state:
			act(env, currentAction, reward);
			cumReward  += reward[1];
			if(!env.game_over()){
				//Obtain active features in the new state:
				Fnext.clear();
				env.getActiveFeaturesIndices(Fnext);
				updateQValues(Fnext, w, Qnext);     //Update Q-values for the new active features
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
			//e <- e + [1 - alpha * e^T phi(S,A)] phi(S,A)
			updateTrace(currentAction, norm_a);
			//theta <- theta + alpha * delta * e + alpha * delta_q (e - phi(S,A))
			updateWeights(currentAction, norm_a, delta_q);
			//e <- gamma * lambda * e
			decayTrace();

			F = Fnext;
			currentAction = nextAction;
		}
		gettimeofday(&tvEnd, NULL);
		timeval_subtract(&tvDiff, &tvEnd, &tvBegin);
		elapsedTime = float(tvDiff.tv_sec) + float(tvDiff.tv_usec)/1000000.0;
		
		float fps = float(env.getEpisodeFrameNumber())/elapsedTime;
		printf("episode: %d,\t%.0f points,\tavg. return: %.1f,\t%d frames,\t%.0f fps\n",
			episode + 1, cumReward - prevCumReward, (float)cumReward/(episode + 1.0),
			env.getEpisodeFrameNumber(), fps);
		totalNumberFrames += env.getEpisodeFrameNumber();
		prevCumReward = cumReward;
		env.reset_game();
	}
}

double TrueOnlineSarsaLearner::evaluatePolicy(Environment<bool>& env){
	float reward = 0;
	float cumReward = 0; 
	float prevCumReward = 0;

	//Repeat (for each episode):
	int episode;
	for(episode = 0; episode < this->numEpisodesEval; episode++){
		//Repeat(for each step of episode) until game is over:
		for(int step = 0; !env.game_over() && step < this->episodeLength; step++){
			//Get state and features active on that state:		
			F.clear();
			env.getActiveFeaturesIndices(F);
			updateQValues(F, w, Q);       //Update Q-values for each possible action
			currentAction = epsilonGreedy(Q);
			//Take action, observe reward and next state:
			reward = env.act(this->actions[currentAction]);
			cumReward  += reward;
		}
		env.reset_game();
		sanityCheck();
		
		printf("%d, %f, %f \n", episode + 1, (float)cumReward/(episode + 1.0), cumReward-prevCumReward);
		
		prevCumReward = cumReward;
	}
	return double(cumReward)/double(episode);
}
