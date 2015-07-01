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
#include <stdio.h>
#include <math.h>

template<typename FeatureType>
TrueOnlineSarsaLearner<FeatureType>::TrueOnlineSarsaLearner(Environment<FeatureType>& env, Parameters *param) : RLLearner<FeatureType>(env, param) {
	delta = 0.0;
	
	alpha = param->getAlpha();
	lambda = param->getLambda();
	traceThreshold = param->getTraceThreshold();
	numFeatures = env.getNumberOfFeatures();

    e.resize(this->numActions);
	for(int i = 0; i < this->numActions; i++){
		//Initialize Q;
		Q.push_back(0);
		Qnext.push_back(0);
		//Initialize e:
		w.push_back(std::vector<double>(numFeatures, 0.0));
	}

	std::stringstream ss;
	ss << "weights_" << param->getSeed() << ".wgt";
	nameWeightsFile =  ss.str();
}


template<typename FeatureType>
TrueOnlineSarsaLearner<FeatureType>::~TrueOnlineSarsaLearner(){}

template<typename FeatureType>
void TrueOnlineSarsaLearner<FeatureType>::updateQValues(std::vector<std::pair<int,FeatureType> > &Features, std::vector<double> &QValues){
	for(int a = 0; a < this->numActions; a++){
		double sumW = 0;
		for(unsigned int i = 0; i < Features.size(); i++){
			sumW += w[a][Features[i].first] * (double)Features[i].second;
		}
		QValues[a] = sumW;
	}
}

template<typename FeatureType>
void TrueOnlineSarsaLearner<FeatureType>::updateWeights(int action, double alpha, double delta_q){
	for(unsigned int a = 0; a < e.size(); a++){
		for(const auto& trace : e[a]){
			int idx = trace.first;
			w[a][idx] = w[a][idx] + alpha * (delta + delta_q) * trace.second;
		}
	}

	for(unsigned int i = 0; i < F.size(); i++){
		int idx = F[i].first;
		w[action][idx] = w[action][idx] - alpha * delta_q * F[i].second;
	}
}

template<typename FeatureType>
void TrueOnlineSarsaLearner<FeatureType>::updateTrace(int action, double alpha){
	double dot_e_phi = 0;
	for(unsigned int i = 0; i < F.size(); i++){
		int idx = F[i].first;
        if(e[action].count(idx)!=0){
            dot_e_phi += e[action][idx]*F[i].second;
        }
	}
	if((1 - alpha * dot_e_phi) > traceThreshold){
		for(unsigned int i = 0; i < F.size(); i++){
			int idx = F[i].first;
            //if the element doesn't exist, we create it
            if(e[action].count(idx) == 0){
                e[action][idx] = 0;
            }
			e[action][idx] = e[action][idx] + (1 - alpha * dot_e_phi)*F[i].second;
		}
	}
}

template<typename FeatureType>
void TrueOnlineSarsaLearner<FeatureType>::decayTrace(){
	//e <- gamma * lambda * e
    for(unsigned int a = 0; a < e.size(); a++){
        for (auto it = e[a].begin(); it != e[a].end() /* not hoisted */; /* no increment */)
        {
            //here it is an iterator on the map. it.first hold the index of the value, and it.second, the value itself
            (*it).second = this->gamma * lambda * (*it).second;
            if ((*it).second < traceThreshold)
            {
                e[a].erase(it++);
            }else{
                ++it;
            }
		}
	}
}

template<typename FeatureType>
void TrueOnlineSarsaLearner<FeatureType>::sanityCheck(){
	for(int i = 0; i < this->numActions; i++){
		if(fabs(Q[i]) > 10e7 || Q[i] != Q[i] /*NaN*/){
			printf("It seems your algorithm diverged!\n");
			exit(0);
		}
	}
}

template<typename FeatureType>
void TrueOnlineSarsaLearner<FeatureType>::dumpWeights(){
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

template<typename FeatureType>
void TrueOnlineSarsaLearner<FeatureType>::loadWeights(){
    std::string line;
	std::ifstream weightsFile (nameWeightsFile.c_str());
	if(weightsFile.is_open()){
		//TODO!!!!
	}
	else{
		printf("Unable to open file to load weights.\n");
	}
}


template<typename FeatureType>
void TrueOnlineSarsaLearner<FeatureType>::learnPolicy(Environment<FeatureType>& env){
	
	struct timeval tvBegin, tvEnd, tvDiff;
	std::vector<double> reward;
	double elapsedTime;
	double norm_a;
	double q_old, delta_q;
	double cumReward = 0, prevCumReward = 0;
	unsigned int maxFeatVectorNorm = 1;
	this->sawFirstReward = 0; this->firstReward = 1.0;

	//Repeat (for each episode):
	int episode, totalNumberFrames = 0;
	//This is going to be interrupted by the ALE code since I set max_num_frames beforehand
	for(episode = 0; totalNumberFrames < this->totalNumberOfFramesToLearn; episode++){ 
		//We have to clean the traces every episode:
		for(unsigned int a = 0; a < e.size(); a++){
            e[a].clear();
		}
		F.clear();

		env.getActiveFeaturesIndices(F);
		updateQValues(F, Q);
		currentAction = this->epsilonGreedy(Q);
		
		q_old = Q[currentAction];

		//Repeat(for each step of episode) until game is over:
		gettimeofday(&tvBegin, NULL);
		//This also stops when the maximum number of steps per episode is reached
		while(!env.game_over()){
			reward.clear();
			reward.push_back(0.0);
			reward.push_back(0.0);
			updateQValues(F, Q);
			sanityCheck();

			//Take action, observe reward and next state:
			this->act(env, currentAction, reward);
			cumReward  += reward[1];
			if(!env.game_over()){
				//Obtain active features in the new state:
				Fnext.clear();
				env.getActiveFeaturesIndices(Fnext);
				updateQValues(Fnext, Qnext);     //Update Q-values for the new active features
				nextAction = this->epsilonGreedy(Qnext);
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
			delta   = reward[0] + this->gamma * Qnext[nextAction] - Q[currentAction];
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
		elapsedTime = double(tvDiff.tv_sec) + double(tvDiff.tv_usec)/1000000.0;
		
		double fps = double(env.getEpisodeFrameNumber())/elapsedTime;
		printf("episode: %d,\t%.0f points,\tavg. return: %.1f,\t%d frames,\t%.0f fps\n",
			episode + 1, cumReward - prevCumReward, (double)cumReward/(episode + 1.0),
			env.getEpisodeFrameNumber(), fps);
		totalNumberFrames += env.getEpisodeFrameNumber();
		prevCumReward = cumReward;
		env.reset_game();
	}
}

template<typename FeatureType>
void TrueOnlineSarsaLearner<FeatureType>::evaluatePolicy(Environment<FeatureType>& env){
	double reward = 0;
	double cumReward = 0; 
	double prevCumReward = 0;

	//Repeat (for each episode):
	for(int episode = 0; episode < this->numEpisodesEval; episode++){
		//Repeat(for each step of episode) until game is over:
		for(int step = 0; !env.game_over() && step < this->episodeLength; step++){
			//Get state and features active on that state:		
			F.clear();
			env.getActiveFeaturesIndices(F);
			updateQValues(F, Q);       //Update Q-values for each possible action
			currentAction = this->epsilonGreedy(Q);
			//Take action, observe reward and next state:
			reward = env.act(this->actions[currentAction]);
			cumReward  += reward;
		}
		env.reset_game();
		sanityCheck();
		
		printf("%d, %f, %f \n", episode + 1, (double)cumReward/(episode + 1.0), cumReward-prevCumReward);
		
		prevCumReward = cumReward;
	}
}
