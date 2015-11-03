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
#include "FastSarsaLearner.hpp"

#include <queue>
#include <stdio.h>
#include <math.h>

using namespace std;

FastSarsaLearner::FastSarsaLearner(Environment<bool>& env, Parameters *param, int seed) : RLLearner<bool>(env, param, seed) {
    totalNumberFrames = 0.0;
    maxFeatVectorNorm = 1;
	
	delta = 0.0;
	kBound = 40;
	alpha = param->getAlpha();
	lambda = param->getLambda();
	traceThreshold = param->getTraceThreshold();
	numFeatures = env.getNumberOfFeatures();
	toSaveCheckPoint = param->getToSaveCheckPoint();
	saveWeightsEveryXSteps = param->getFrequencySavingWeights();
	pathWeightsFileToLoad = param->getPathToWeightsFiles();
	featureSeen.resize(numActions);
	numEpisodesEval = param->getNumEpisodesEval();
	numEpisodesLearn = param->getNumEpisodesLearn();
	
	for(int i = 0; i < numActions; i++){
		//Initialize Q;
		Q.push_back(0);
		Qnext.push_back(0);

		w.push_back(vector<float>(numFeatures, 0.0));
		nonZeroElig.push_back(vector<int>());
	}

    episodePassed = 0;
	if(toSaveCheckPoint){
        checkPointName = param->getCheckPointName();
        //load CheckPoint
        ifstream checkPointToLoad;
        string checkPointLoadName = checkPointName+"-checkPoint.txt";
        checkPointToLoad.open(checkPointLoadName.c_str());
        if (checkPointToLoad.is_open()){
            loadCheckPoint(checkPointToLoad);
            remove(checkPointLoadName.c_str());
        }
        ofstream learningConditionFile;
        nameForLearningCondition = checkPointName+"-learningCondition-Episode"+to_string(episodePassed)+"-finished.txt";
        string previousNameForLearningCondition =checkPointName +"-learningCondition.txt";
        rename(previousNameForLearningCondition.c_str(), nameForLearningCondition.c_str());
        learningConditionFile.open(nameForLearningCondition, ios_base::app);
        learningConditionFile.close();
    }
}

FastSarsaLearner::~FastSarsaLearner(){}

void FastSarsaLearner::sanityCheck(){
	for(int i = 0; i < numActions; i++){
		if(fabs(Q[i]) > 10e7 || Q[i] != Q[i] /*NaN*/){
			printf("It seems your algorithm diverged!\n");
			exit(0);
		}
	}
}

//To do: we do not want to save weights that are zero
void FastSarsaLearner::saveCheckPoint(int episode, int totalNumberFrames, vector<float>& episodeResults,int& frequency,vector<int>& episodeFrames, vector<double>& episodeFps){
    ofstream learningConditionFile;
    string newNameForLearningCondition = checkPointName+"-learningCondition-Episode"+to_string(episode)+"-writing.txt";
    int renameReturnCode = rename(nameForLearningCondition.c_str(),newNameForLearningCondition.c_str());
    if (renameReturnCode == 0){
        nameForLearningCondition = newNameForLearningCondition;
        learningConditionFile.open(nameForLearningCondition.c_str(), ios_base::app);
        int numEpisode = episodeResults.size();
        for (int index = 0;index<numEpisode;index++){
            learningConditionFile <<"Episode "<<episode-numEpisode+1+index<<": "<<episodeResults[index]<<" points,  "<<episodeFrames[index]<<" frames,  "<<episodeFps[index]<<" fps."<<endl;
        }
        episodeResults.clear();
        episodeFrames.clear();
        episodeFps.clear();
        learningConditionFile.close();
        newNameForLearningCondition.replace(newNameForLearningCondition.end()-11,newNameForLearningCondition.end()-4,"finished");
        rename(nameForLearningCondition.c_str(),newNameForLearningCondition.c_str());
        nameForLearningCondition = newNameForLearningCondition;
    }
    
    //write parameters checkPoint
    string currentCheckPointName = checkPointName+"-checkPoint-Episode"+to_string(episode)+"-writing.txt";
    ofstream checkPointFile;
    checkPointFile.open(currentCheckPointName.c_str());
    checkPointFile<<agentRand<<endl;
    checkPointFile<<totalNumberFrames<<endl;
    checkPointFile << episode<<endl;
    checkPointFile << firstReward<<endl;
    checkPointFile << maxFeatVectorNorm<<endl;
    for(int a = 0; a < featureSeen.size(); a++){
        for(int index = 0; index < featureSeen[a].size(); index++){
            checkPointFile << a << " " << featureSeen[a][index] << " " << w[a][featureSeen[a][index]] << "\t";
        }
    }
    checkPointFile << endl;
    checkPointFile.close();
    string previousVersionCheckPoint = checkPointName+"-checkPoint-Episode"+to_string(episode-frequency)+"-finished.txt";
    remove(previousVersionCheckPoint.c_str());
    string oldCheckPointName = currentCheckPointName;
    currentCheckPointName.replace(currentCheckPointName.end()-11,currentCheckPointName.end()-4,"finished");
    rename(oldCheckPointName.c_str(),currentCheckPointName.c_str());
}

void FastSarsaLearner::loadCheckPoint(ifstream& checkPointToLoad){
    checkPointToLoad >> agentRand;
    checkPointToLoad >> totalNumberFrames;
    checkPointToLoad >> episodePassed;
    checkPointToLoad >> firstReward;
    checkPointToLoad >> maxFeatVectorNorm;
    int action, index;
    float weight;
    while (checkPointToLoad>>action && checkPointToLoad>>index && checkPointToLoad>>weight){
        w[action][index] = weight;
    }
    checkPointToLoad.close();
}

void FastSarsaLearner::learnPolicy(Environment<bool>& env){
	
	struct timeval tvBegin, tvEnd, tvDiff;
	vector<float> reward;
	float elapsedTime;
	float cumReward = 0, prevCumReward = 0;
	sawFirstReward = 0; firstReward = 1.0;
	vector<float> episodeResults;
    vector<int> episodeFrames;
    vector<double> episodeFps;

	//Repeat (for each episode):
	for(int episode = episodePassed + 1; episode <= numEpisodesLearn /*totalNumberFrames < totalNumberOfFramesToLearn*/; episode++){
		queue<vector<int>> nuFeatures;
		queue<float> nuDelta;
		queue<int> nuActions;

		int step = 0;
		float u = 0.0;
		float uSync = 0.0;
		float vCurrent = 0.0;
		float vNext = 0.0;
		bool ready = false;
		
		F.clear();
		env.getActiveFeaturesIndices(F);

		updateQValues(F, w, Q);
		currentAction = epsilonGreedy(Q);

		//Repeat(for each step of episode) until game is over:
		gettimeofday(&tvBegin, NULL);

		//This also stops when the maximum number of steps per episode is reached
		while(!env.isTerminal()){
			reward.clear();
			reward.push_back(0.0);
			reward.push_back(0.0);
			updateQValues(F, w, Q);
			sanityCheck();

			//Take action, observe reward and next state:
			act(env, currentAction, reward);
			cumReward  += reward[1];

			if(!env.isTerminal()){
				//Obtain active features in the new state:
				Fnext.clear();
				env.getActiveFeaturesIndices(Fnext);

				updateQValues(Fnext, w, Qnext);     //Update Q-values for the new active features

				nextAction = epsilonGreedy(Qnext);
				vNext = Qnext[nextAction];

			}
			else{
				vNext = 0.0;
			}
			//To ensure the learning rate will never increase along
			//the time, Marc used such approach in his JAIR paper		
			if (F.size() > maxFeatVectorNorm){
				maxFeatVectorNorm = F.size();
			}

			delta = reward[0] + gamma * (1 - lambda) * vNext;
			nuFeatures.push(F);
			nuDelta.push(delta);
			nuActions.push(currentAction);

			delta = reward[0] + gamma * vNext - vCurrent;
			vCurrent = vNext;

			if(step == kBound - 1){
				u = uSync;
				uSync = vCurrent;
				step = 0;
				ready = true;
			} else{
				step = step + 1;
				uSync = uSync + pow(gamma * lambda, step - 1) * delta;
			}

			if(ready){
				vector<int> F_queue = nuFeatures.front();
				nuFeatures.pop();
				float delta_queue = nuDelta.front();
				nuDelta.pop();
				int action_queue = nuActions.front();
				nuActions.pop();

				u = u + pow(gamma * lambda, kBound - 1) * delta;

				updateQValues(F_queue, w, Q);
				for(int i = 0; i < F_queue.size(); i++){
					int idx = F_queue[i];
					delta = u - Q[action_queue];
					if (w[action_queue][idx] == 0 && delta != 0){
                        featureSeen[action_queue].push_back(idx);
                    }
					w[action_queue][F_queue[i]] += (alpha/maxFeatVectorNorm) * delta;
				}
				u = (u - delta_queue)/(gamma * lambda);
			}
			F = Fnext;
			currentAction = nextAction;
		}

		if(!ready){
			u = uSync;
		}
		while(!nuFeatures.empty()){
			vector<int> F_queue = nuFeatures.front();
			nuFeatures.pop();
			float delta_queue = nuDelta.front();
			nuDelta.pop();
			int action_queue = nuActions.front();
			nuActions.pop();

			updateQValues(F_queue, w, Q);
			for(int i = 0; i < F_queue.size(); i++){
				int idx = F_queue[i];
				delta = u - Q[action_queue];
				if (w[action_queue][idx] == 0 && delta != 0){
                    featureSeen[action_queue].push_back(idx);
                }
				w[action_queue][F_queue[i]] += (alpha/maxFeatVectorNorm) * delta;
			}
			u = (u - delta_queue)/(gamma * lambda);
		}

		gettimeofday(&tvEnd, NULL);
		timeval_subtract(&tvDiff, &tvEnd, &tvBegin);
		elapsedTime = float(tvDiff.tv_sec) + float(tvDiff.tv_usec)/1000000.0;
		
		float fps = float(env.getEpisodeFrameNumber())/elapsedTime;
		printf("episode: %d,\t%.0f points,\tavg. return: %.1f,\t%d frames,\t%.0f fps\n",
			episode, cumReward - prevCumReward, (float)cumReward/(float) episode,
			env.getEpisodeFrameNumber(), fps);
        episodeResults.push_back(cumReward-prevCumReward);
        episodeFrames.push_back(env.getEpisodeFrameNumber());
        episodeFps.push_back(fps);
		totalNumberFrames += env.getEpisodeFrameNumber();
		prevCumReward = cumReward;
//		rand(); //I'm calling this to have equivalence to Harm's code.
		env.reset_game();
		if(toSaveCheckPoint && episode%saveWeightsEveryXSteps == 0){
            saveCheckPoint(episode,totalNumberFrames,episodeResults,saveWeightsEveryXSteps,episodeFrames,episodeFps);
        }
	}
}

double FastSarsaLearner::evaluatePolicy(Environment<bool>& env){
	float reward = 0;
	float cumReward = 0; 
	float prevCumReward = 0;
	struct timeval tvBegin, tvEnd, tvDiff;
	float elapsedTime;

    std::string oldName = checkPointName+"-Result-writing.txt";
    std::string newName = checkPointName+"-Result-finished.txt";
    std::ofstream resultFile;
    resultFile.open(oldName.c_str());

	//Repeat (for each episode):
	int episode;
	for(episode = 1; episode <= numEpisodesEval; episode++){
		//Repeat(for each step of episode) until game is over:
		for(int step = 0; !env.isTerminal() && step < episodeLength; step++){
			//Get state and features active on that state:		
			F.clear();
			env.getActiveFeaturesIndices(F);
			updateQValues(F, w, Q);       //Update Q-values for each possible action
			currentAction = epsilonGreedy(Q);
			//Take action, observe reward and next state:
			reward = env.act(actions[currentAction]);
			cumReward  += reward;
		}
		gettimeofday(&tvEnd, NULL);
		timeval_subtract(&tvDiff, &tvEnd, &tvBegin);
		elapsedTime = float(tvDiff.tv_sec) + float(tvDiff.tv_usec)/1000000.0;
		float fps = float(env.getEpisodeFrameNumber())/elapsedTime;

		resultFile << "Episode " << episode << ": " << cumReward - prevCumReward << std::endl;
		printf("episode: %d,\t%.0f points,\tavg. return: %.1f,\t%d frames,\t%.0f fps\n", 
			episode, (cumReward-prevCumReward), (float)cumReward/(float) episode, env.getEpisodeFrameNumber(), fps);

		env.reset();
		prevCumReward = cumReward;
	}
	resultFile << "Average: " << (double)cumReward/numEpisodesEval << std::endl;
    resultFile.close();
    rename(oldName.c_str(),newName.c_str());

    return double(cumReward)/double(episode);
}

void FastSarsaLearner::saveWeightsToFile(string suffix){
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

void FastSarsaLearner::loadWeights(){
    string line;
    int nActions, nFeatures;
    int i, j;
    double value;
    
    std::ifstream weightsFile (pathWeightsFileToLoad.c_str());
    
    weightsFile >> nActions >> nFeatures;
    assert(nActions == numActions);
    assert(nFeatures == numFeatures);
    
    while(weightsFile >> i >> j >> value){
        w[i][j] = value;
    }
}
