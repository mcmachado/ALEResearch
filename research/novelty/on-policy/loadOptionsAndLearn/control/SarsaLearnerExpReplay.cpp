/****************************************************************************************
** Implementation of Sarsa(lambda). It implements Fig. 8.8 (Linear, gradient-descent   **
** Sarsa(lambda)) from the book "R. Sutton and A. Barto; Reinforcement Learning: An    ** 
** Introduction. 1st edition. 1988."                                                   **
** Some updates are made to make it more efficient, as not iterating over all features.**
**                                                                                     **
** Author: Marlos C. Machado                                                           **
****************************************************************************************/

#include <stdio.h>
#include <math.h>

#include "SarsaLearnerExpReplay.hpp"
#include "../../../../../src/common/Timer.hpp"
#include "../../../../../src/common/Mathematics.hpp"

using namespace std;

SarsaExpReplay::SarsaExpReplay(ALEInterface& ale, Features *features, Parameters *param) : RLLearner(ale, features, param) {
	delta = 0.0;

	alpha = param->getAlpha();
	lambda = param->getLambda();
	traceThreshold = param->getTraceThreshold();
	numFeatures = features->getNumberOfFeatures();
	toSaveWeightsAfterLearning = param->getToSaveWeightsAfterLearning();
	saveWeightsEveryXSteps = param->getFrequencySavingWeights();
	pathWeightsFileToLoad = param->getPathToWeightsFiles();
	
	for(int i = 0; i < numBasicActions; i++){
		//Initialize Q;
		Q.push_back(0);
		Qnext.push_back(0);
		//Initialize e:
		e.push_back(vector<float>(numFeatures, 0.0));
		w.push_back(vector<float>(numFeatures, 0.0));
		nonZeroElig.push_back(vector<int>());
	}

	if(toSaveWeightsAfterLearning){
		std::stringstream ss;
		ss << param->getFileWithWeights() << param->getSeed() << ".wgt";
		nameWeightsFile =  ss.str();
	}

	if(param->getToLoadWeights()){
		loadWeights();
	}
}

SarsaExpReplay::~SarsaExpReplay(){}

void SarsaExpReplay::updateQValues(vector<int> &Features, vector<float> &QValues){
	for(int a = 0; a < QValues.size(); a++){
		float sumW = 0;
		for(unsigned int i = 0; i < Features.size(); i++){
			sumW += w[a][Features[i]];
		}
		QValues[a] = sumW;
	}
}

void SarsaExpReplay::updateReplTrace(int action, vector<int> &Features){
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

void SarsaExpReplay::sanityCheck(){
	for(int i = 0; i < Q.size(); i++){
		if(fabs(Q[i]) > 10e7 || Q[i] != Q[i] /*NaN*/){
			printf("It seems your algorithm diverged!\n");
			exit(0);
		}
	}
}

void SarsaExpReplay::saveWeightsToFile(string suffix){
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

void SarsaExpReplay::loadWeights(){
	string line;
	int nActions, nFeatures;
	int i, j;
	float value;

	std::ifstream weightsFile (pathWeightsFileToLoad.c_str());
	
	weightsFile >> nActions >> nFeatures;
	assert(nActions == numBasicActions);
	assert(nFeatures == numFeatures);

	while(weightsFile >> i >> j >> value){
		w[i][j] = value;
	}
}

int SarsaExpReplay::actionFromOptions(vector<int> &Features, vector<vector<vector<float> > > &learnedOptions){
	float termProb = 0.0;
	int nextAction = -1;
	int numActionsInOption = -1;
	if(optionBeingPlayed.size() != 0){
		int currentOption = optionBeingPlayed[optionBeingPlayed.size() - 1];
		numActionsInOption = learnedOptions[currentOption].size();
		//printf("playing option %d\n", currentOption);
		//We need to be epsilon-greedy w.r.t. the argmax:
		vector<float> Q(numActionsInOption, 0.0);
		for(int a = 0; a < Q.size(); a++){
			float sumW = 0;
			for(unsigned int i = 0; i < Features.size(); i++){
				sumW += learnedOptions[currentOption][a][Features[i]];
			}
			Q[a] = sumW;
		}
		nextAction = epsilonGreedy(Q);
	} else{ //If we did not commit to an action yet:
		nextAction = rand() % numTotalActions;
	}
	
	//What if the next action is an option?
	if(nextAction >= numBasicActions){
		optionBeingPlayed.push_back(nextAction - numBasicActions);
		nextAction = actionFromOptions(Features, learnedOptions);
	}

	if(numActionsInOption > numBasicActions){
		termProb = 0.20;
	} else{
		termProb = 0.05;
	}

	if(rand()%1000 < 1000 * termProb){
		if(optionBeingPlayed.size() != 0){
			int currentOption = optionBeingPlayed[optionBeingPlayed.size() - 1];
			//printf("stopping option %d\n", currentOption);
			optionBeingPlayed.pop_back();
		}
	}

	return nextAction;
}

int SarsaExpReplay::getNextAction(vector<int> &Features, vector<float> &QValues, 
	int episode, vector<vector<vector<float> > > &learnedOptions){
	int nextAction = -1;
	if(episode % 10 < 5){
		nextAction = actionFromOptions(Features, learnedOptions);
	}
	else{
		nextAction = epsilonGreedy(QValues);
	}
	return nextAction;
}

void SarsaExpReplay::learnPolicy(ALEInterface& ale, Features *features, vector<vector<vector<float> > > &learnedOptions){
	struct timeval tvBegin, tvEnd, tvDiff;
	vector<float> reward;
	float elapsedTime;
	float cumReward = 0, prevCumReward = 0;
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
		//currentAction = getNextAction(F, Q, episode, learnedOptions);
		gettimeofday(&tvBegin, NULL);

		//This also stops when the maximum number of steps per episode is reached
		while(!ale.game_over()){
			reward.clear();
			reward.push_back(0.0);
			reward.push_back(0.0);
			updateQValues(F, Q);
			sanityCheck();

			//Take action, observe reward and next state:
			currentAction = getNextAction(F, Q, episode, learnedOptions);
			act(ale, currentAction, features, reward, learnedOptions, w);
			cumReward  += reward[1];

			if(!ale.game_over()){
				//Obtain active features in the new state:
				Fnext.clear();
				features->getActiveFeaturesIndices(ale.getScreen(), ale.getRAM(), Fnext);
				updateQValues(Fnext, Qnext);     //Update Q-values for the new active features
				//nextAction = getNextAction(F, Q, episode, learnedOptions);
				nextAction = Mathematics::argmax(Qnext);
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
				updateReplTrace(currentAction, F);
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
					w[a][idx] = w[a][idx] + (alpha/maxFeatVectorNorm) * delta * e[a][idx];
				}
			}
			F = Fnext;
		}
		gettimeofday(&tvEnd, NULL);
		timeval_subtract(&tvDiff, &tvEnd, &tvBegin);
		elapsedTime = float(tvDiff.tv_sec) + float(tvDiff.tv_usec)/1000000.0;
		
		float fps = float(ale.getEpisodeFrameNumber())/elapsedTime;
		if(episode % 10 < 5){
			printf("*");
		}
		printf("episode: %d,\t%.0f points,\tavg. return: %.1f,\t%d frames,\t%.0f fps\n",
			episode + 1, cumReward - prevCumReward, (float)cumReward/(episode + 1.0),
			ale.getEpisodeFrameNumber(), fps);
		totalNumberFrames += ale.getEpisodeFrameNumber();
		prevCumReward = cumReward;
		ale.reset_game();
		if(toSaveWeightsAfterLearning && episode%saveWeightsEveryXSteps == 0 && episode > 0){
			stringstream ss;
			ss << episode;
			saveWeightsToFile(ss.str());
		}
	}
	if(toSaveWeightsAfterLearning){
		stringstream ss;
		ss << episode;
		saveWeightsToFile(ss.str());
	}
}

void SarsaExpReplay::evaluatePolicy(ALEInterface& ale, Features *features){
	/* This code does not play options. TODO: Fix it.
	float reward = 0;
	float cumReward = 0; 
	float prevCumReward = 0;
	struct timeval tvBegin, tvEnd, tvDiff;
	float elapsedTime;

	//Repeat (for each episode):
	for(int episode = 0; episode < numEpisodesEval; episode++){
		gettimeofday(&tvBegin, NULL);
		//Repeat(for each step of episode) until game is over:
		for(int step = 0; !ale.game_over() && step < episodeLength; step++){
			//Get state and features active on that state:		
			F.clear();
			features->getActiveFeaturesIndices(ale.getScreen(), ale.getRAM(), F);
			updateQValues(F, Q);       //Update Q-values for each possible action
			currentAction = epsilonGreedy(Q);
			//Take action, observe reward and next state:
			reward = 0;
			reward += ale.act(actions[currentAction]);
			cumReward  += reward;
		}
		gettimeofday(&tvEnd, NULL);
		timeval_subtract(&tvDiff, &tvEnd, &tvBegin);
		elapsedTime = float(tvDiff.tv_sec) + float(tvDiff.tv_usec)/1000000.0;
		float fps = float(ale.getEpisodeFrameNumber())/elapsedTime;

		printf("episode: %d,\t%.0f points,\tavg. return: %.1f,\t%d frames,\t%.0f fps\n", 
			episode + 1, (cumReward-prevCumReward), (float)cumReward/(episode + 1.0), ale.getEpisodeFrameNumber(), fps);

		ale.reset_game();
		prevCumReward = cumReward;
	}
	*/
}
