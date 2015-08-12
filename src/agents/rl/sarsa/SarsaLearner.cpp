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
#include "SarsaLearner.hpp"
#include <stdio.h>
#include <math.h>

int rr;

SarsaLearner::SarsaLearner(Environment<bool>& env, Parameters *param) : RLLearner<bool>(env, param) {
	delta = 0.0;
    
	alpha = param->getAlpha();
	lambda = param->getLambda();
	traceThreshold = param->getTraceThreshold();
	numFeatures = env.getNumberOfFeatures();
	toSaveWeightsAfterLearning = param->getToSaveWeightsAfterLearning();
	saveWeightsEveryXSteps = param->getFrequencySavingWeights();
	pathWeightsFileToLoad = param->getPathToWeightsFiles();

    e.resize(numActions);
	for(int i = 0; i < numActions; i++){
		//Initialize Q;
		Q.push_back(0);
		Qnext.push_back(0);
		//Initialize e:
		w.push_back(std::vector<float>(numFeatures, 0.0));
	}

	if(toSaveWeightsAfterLearning){
		std::stringstream ss;
		ss << param->getFileWithWeights() << param->getSeed() << ".wgt";
		nameWeightsFile =  ss.str();
	}

	if(param->getToLoadWeights()){
		loadWeights();
	}

    rr=-1;
}

SarsaLearner::~SarsaLearner(){}

void SarsaLearner::updateQValues(std::vector<int> &Features, std::vector<float> &QValues){
	for(int a = 0; a < numActions; a++){
		float sumW = 0;
		for(unsigned int i = 0; i < Features.size(); i++){
			sumW += w[a][Features[i]];
		}
		QValues[a] = sumW;
	}
}
void SarsaLearner::updateBehaviorQValues(std::vector<int> &Features, std::vector<float> &QValues,const std::vector<std::vector<float>>& weights)
{
	for(int a = 0; a < numActions; a++){
		float sumW = 0;
		for(unsigned int i = 0; i < Features.size(); i++){
			sumW += weights[a][Features[i]];
		}
		QValues[a] = sumW;
	}
}

void SarsaLearner::updateReplTrace(int action, std::vector<int> &Features){
	//e <- gamma * lambda * e
    for(unsigned int a = 0; a < e.size(); a++){
        for (auto it = e[a].begin(); it != e[a].end() /* not hoisted */; /* no increment */)
        {
            //here it is an iterator on the map. it.first hold the index of the value, and it.second, the value itself
            (*it).second = gamma * lambda * (*it).second;
            if ((*it).second < traceThreshold)
            {
                e[a].erase(it++);
            }else{
                ++it;
            }
		}
	}
	//For all i in Fa:
	for(unsigned int i = 0; i < F.size(); i++){
		int idx = Features[i];
        //if the element doesn't exist, we create it
        if(e[action].count(idx) == 0){
            e[action][idx] = 0;
        }
		e[action][idx] = 1;
	}
}

void SarsaLearner::updateAcumTrace(int action, std::vector<int> &Features){
	//e <- gamma * lambda * e
	for(unsigned int a = 0; a < e.size(); a++){
        for (auto it = e[a].begin(); it != e[a].end() /* not hoisted */; /* no increment */)
        {
            //here it is an iterator on the map. it.first hold the index of the value, and it.second, the value itself
            (*it).second = gamma * lambda * (*it).second;
            if ((*it).second < traceThreshold)
            {
                e[a].erase(it++);
            }else{
                ++it;
            }
		}
	}

	//For all i in Fa:
	for(unsigned int i = 0; i < F.size(); i++){
		int idx = Features[i];
        //if the element doesn't exist, we create it
        if(e[action].count(idx) == 0){
            e[action][idx] = 0;
        }
		e[action][idx] += 1;
	}
}

void SarsaLearner::sanityCheck(){
	for(int i = 0; i < numActions; i++){
		if(fabs(Q[i]) > 10e7 || Q[i] != Q[i] /*NaN*/){
			printf("It seems your algorithm diverged!\n");
			exit(0);
		}
	}
}

void SarsaLearner::saveWeightsToFile(std::string suffix){
	std::ofstream weightsFile ((nameWeightsFile + suffix).c_str());
	if(weightsFile.is_open()){
		weightsFile << w.size() << " " << w[0].size() << std::endl;
		for(unsigned int i = 0; i < w.size(); i++){
			for(unsigned int j = 0; j < w[i].size(); j++){
				if(w[i][j] != 0){
					weightsFile << i << " " << j << " " << w[i][j] << std::endl;
				}
			}
		}
		weightsFile.close();
	}
	else{
		printf("Unable to open file to write weights.\n");
	}
}

void SarsaLearner::loadWeights(std::string fname){
   std::string line;
	int nActions, nFeatures;
	int i, j;
	float value;

	std::ifstream weightsFile (fname.c_str());
	
	weightsFile >> nActions >> nFeatures;
	assert(nActions == numActions);
	assert(nFeatures == numFeatures);

	while(weightsFile >> i >> j >> value){
		w[i][j] = value;
	}
}
void SarsaLearner::loadWeights(){
    loadWeights(pathWeightsFileToLoad.c_str());
}

void SarsaLearner::learnPolicy(Environment<bool>& env, bool transfering,const std::vector<std::vector<float> >& b_policy){
	
	struct timeval tvBegin, tvEnd, tvDiff;
    std::vector<double> reward;
	double elapsedTime;
	float cumReward = 0, prevCumReward = 0;
	unsigned int maxFeatVectorNorm = 1;
	sawFirstReward = 0; firstReward = 1.0;
    std::vector<float> bQ(numActions,0.0);               //Q(a) entries for behavior policy

	//Repeat (for each episode):
	int episode, totalNumberFrames = 0;
	//This is going to be interrupted by the ALE code since I set max_num_frames beforehand
    rr=1;
	for(episode = 0; totalNumberFrames < totalNumberOfFramesToLearn; episode++){ 
		//We have to clean the traces every episode:
		for(unsigned int a = 0; a < e.size(); a++){
            e[a].clear();
		}
		F.clear();

		env.getActiveFeaturesIndices(F);
		updateQValues(F, Q);
        if(transfering){		
            updateBehaviorQValues(F,bQ,b_policy);
            currentAction = epsilonGreedy(bQ);
        }else{
            currentAction = epsilonGreedy(Q);
        }
		//Repeat(for each step of episode) until game is over:
		gettimeofday(&tvBegin, NULL);

		//This also stops when the maximum number of steps per episode is reached
		while(!env.isTerminal()){
			reward.clear();
			reward.push_back(0.0);
			reward.push_back(0.0);
			updateQValues(F, Q);

			sanityCheck();
			//Take action, observe reward and next state:
			act(env, currentAction, reward);
			cumReward  += reward[1];
			if(!env.isTerminal()){
				//Obtain active features in the new state:
				Fnext.clear();
				env.getActiveFeaturesIndices(Fnext);
				updateQValues(Fnext, Qnext);     //Update Q-values for the new active features
                if(transfering){		
                    updateBehaviorQValues(Fnext,bQ,b_policy);
                    nextAction = epsilonGreedy(bQ);
                }else{
                    nextAction = epsilonGreedy(Qnext);
                }
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

			delta = reward[0] + gamma * Qnext[nextAction] - Q[currentAction];

			updateReplTrace(currentAction, F);
			//Update weights vector:
			for(unsigned int a = 0; a < e.size(); a++){
                for(const auto& trace : e[a]){
                    int idx = trace.first;
					w[a][idx] = w[a][idx] + (alpha/maxFeatVectorNorm) * delta * trace.second;
				}
			}
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
        if(totalNumberFrames > rr*2000000.0/double(numFlavors)){
            double perf=evaluatePolicy(env);
            std::cout<<"rr "<<rr<<" frame "<<totalNumberFrames<<" perf "<<perf<<std::endl;
            rr++;
        }
		prevCumReward = cumReward;
		env.reset();
		if(toSaveWeightsAfterLearning && episode%saveWeightsEveryXSteps == 0 && episode > 0){
            std::stringstream ss;
			ss << episode;
			saveWeightsToFile(ss.str());
		}
	}
	if(toSaveWeightsAfterLearning){
        std::stringstream ss;
		ss << episode;
		saveWeightsToFile(ss.str());
	}
    rr=-1;
}

double SarsaLearner::evaluatePolicy(Environment<bool>& env){
    return evaluatePolicy(env,numEpisodesEval);
}
double SarsaLearner::evaluatePolicy(Environment<bool>& env,unsigned numSteps, bool epsilonAnneal){
	float reward = 0;
	float cumReward = 0; 
	float prevCumReward = 0;
	struct timeval tvBegin, tvEnd, tvDiff;
	double elapsedTime;
    if(epsilonAnneal)
        epsilon = 1.0;
	//Repeat (for each episode):
	for(int episode = 0; episode < numSteps; episode++){
		//Repeat(for each step of episode) until game is over:
		for(int step = 0; !env.isTerminal() && step < episodeLength; step++){
			//Get state and features active on that state:		
			F.clear();
			env.getActiveFeaturesIndices(F);
			updateQValues(F, Q);       //Update Q-values for each possible action
			currentAction = epsilonGreedy(Q);
            //compute proba of taking current action
            //first, we need the number of QValues that are tied
            double numTies = 0;
            if(!randomActionTaken){
                for(const auto& q : Q){
                    if(q==Q[currentAction])
                        numTies++;
                }
            }
            
            double proba_action = epsilon/double(numActions) + (randomActionTaken ? 0 : (1.0 - epsilon)/numTies);
			//Take action, observe reward and next state:
			reward = env.act(actions[currentAction],proba_action);
			cumReward  += reward;
		}
		gettimeofday(&tvEnd, NULL);
		timeval_subtract(&tvDiff, &tvEnd, &tvBegin);
		elapsedTime = double(tvDiff.tv_sec) + double(tvDiff.tv_usec)/1000000.0;
		double fps = double(env.getEpisodeFrameNumber())/elapsedTime;
        if(rr==-1)
            printf("episode: %d,\t%.0f points,\tavg. return: %.1f,\t%d frames,\t%.0f fps epsilon : %f\n", 
                   episode + 1, (cumReward-prevCumReward), (double)cumReward/(episode + 1.0), env.getEpisodeFrameNumber(), fps,epsilon);

		env.reset();
		prevCumReward = cumReward;
        if(epsilonAnneal)
            epsilon-=1/(double)(numSteps);
	}
    return cumReward/(double)(numSteps);
}



