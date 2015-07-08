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
#include "../../../common/Timer.hpp"
#endif
#include "SarsaLearner.hpp"
#include <stdio.h>
#include <math.h>
using namespace std;

SarsaLearner::SarsaLearner(ALEInterface& ale, Features *features, Parameters *param,int seed) : RLLearner(ale, param,seed) {
    totalNumberFrames = 0.0;
    maxFeatVectorNorm = 1;
    saveThreshold =0;
    
    delta = 0.0;
	alpha = param->getAlpha();
	lambda = param->getLambda();
	traceThreshold = param->getTraceThreshold();
	numFeatures = features->getNumberOfFeatures();
    toSaveCheckPoint = param->getToSaveCheckPoint();
    saveWeightsEveryXFrames = param->getFrequencySavingWeights();
	pathWeightsFileToLoad = param->getPathToWeightsFiles();
    //featureSeen.resize(numActions);
	
	for(int i = 0; i < numActions; i++){
		//Initialize Q;
		Q.push_back(0);
		Qnext.push_back(0);
		//Initialize e:
		e.push_back(unordered_map<long long, float>());
		w.push_back(unordered_map<long long,float>());
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
        saveThreshold = (totalNumberFrames/saveWeightsEveryXFrames)*saveWeightsEveryXFrames;
        ofstream learningConditionFile;
        nameForLearningCondition = checkPointName+"-learningCondition-Frames"+to_string(saveThreshold)+"-finished.txt";
        string previousNameForLearningCondition =checkPointName +"-learningCondition.txt";
        rename(previousNameForLearningCondition.c_str(), nameForLearningCondition.c_str());
        saveThreshold+=saveWeightsEveryXFrames;
        learningConditionFile.open(nameForLearningCondition, ios_base::app);
        learningConditionFile.close();

    }
}

SarsaLearner::~SarsaLearner(){}

void SarsaLearner::updateQValues(vector<long long> &Features, vector<float> &QValues){
	for(int a = 0; a < numActions; a++){
		float sumW = 0;
		for(unsigned long long i = 0; i < Features.size(); i++){
            auto got = w[a].find(Features[i]);
            if (got!=w[a].end()){
                sumW += w[a][Features[i]];
            }
		}
		QValues[a] = sumW;
	}
}

void SarsaLearner::updateReplTrace(int action, vector<long long> &Features){
	//e <- gamma * lambda * e
	for(unsigned int a = 0; a < e.size(); a++){
        for (auto it=e[a].begin();it!=e[a].end();){
            it->second = it->second*gamma*lambda;
            if (it->second<traceThreshold){
                it = e[a].erase(it);
            }else{
                it++;
            }
        }
    }
	//For all i in Fa:
	for(unsigned int i = 0; i < F.size(); i++){
		long long idx = Features[i];
		//If the trace is zero it is not in the vector
		//of non-zeros, thus it needs to be added
		e[action][idx] = 1;
	}
}

void SarsaLearner::updateAcumTrace(int action, vector<long long> &Features){
	//e <- gamma * lambda * e
    for(unsigned int a = 0; a < e.size(); a++){
        for (auto it=e[a].begin();it!=e[a].end();){
            it->second = gamma*lambda;
            if (it->second<traceThreshold){
                it = e[a].erase(it);
            }else{
                it++;
            }
        }
    }
	//For all i in Fa:
	for(unsigned int i = 0; i < F.size(); i++){
		long long idx = Features[i];
		//If the trace is zero it is not in the vector
		//of non-zeros, thus it needs to be added
        auto got = e[action].find(idx);
        if (got==e[action].end()){
            e[action][idx]=1;
        }else{
            got->second+=1;
        }
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

//To do: we do not want to save weights that are zero
void SarsaLearner::saveCheckPoint(int episode, int totalNumberFrames, vector<float>& episodeResults,int& frequency,vector<int>& episodeFrames, vector<double>& episodeFps){
    ofstream learningConditionFile;
    string newNameForLearningCondition = checkPointName+"-learningCondition-Frames"+to_string(saveThreshold)+"-writing.txt";
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
    string currentCheckPointName = checkPointName+"-checkPoint-Frames"+to_string(saveThreshold)+"-writing.txt";
    ofstream checkPointFile;
    checkPointFile.open(currentCheckPointName.c_str());
    checkPointFile<<agentRand<<endl;
    checkPointFile<<totalNumberFrames<<endl;
    checkPointFile << episode<<endl;
    checkPointFile << firstReward<<endl;
    checkPointFile << maxFeatVectorNorm<<endl;
    for (int a=0;a<w.size();a++){
        for (auto it=w[a].begin(); it!=w[a].end();it++){
            checkPointFile<<a<<" "<<it->first<<" "<<it->second<<"\t";
        }
    }
    
    checkPointFile << endl;
    checkPointFile.close();
    string previousVersionCheckPoint = checkPointName+"-checkPoint-Frames"+to_string(saveThreshold-saveWeightsEveryXFrames)+"-finished.txt";
    remove(previousVersionCheckPoint.c_str());
    string oldCheckPointName = currentCheckPointName;
    currentCheckPointName.replace(currentCheckPointName.end()-11,currentCheckPointName.end()-4,"finished");
    rename(oldCheckPointName.c_str(),currentCheckPointName.c_str());

}

void SarsaLearner::loadCheckPoint(ifstream& checkPointToLoad){
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

void SarsaLearner::learnPolicy(ALEInterface& ale, Features *features){
	
	struct timeval tvBegin, tvEnd, tvDiff;
	vector<float> reward;
	double elapsedTime;
	double cumReward = 0, prevCumReward = 0;
	sawFirstReward = 0; firstReward = 1.0;
    vector<float> episodeResults;
    vector<int> episodeFrames;
    vector<double> episodeFps;

	//Repeat (for each episode):
	//This is going to be interrupted by the ALE code since I set max_num_frames beforehand
    for(int episode = episodePassed+1; totalNumberFrames < totalNumberOfFramesToLearn; episode++){
		//We have to clean the traces every episode:
		for(unsigned int a = 0; a < e.size(); a++){
            e[a].clear();
        }
        
		F.clear();
		features->getActiveFeaturesIndices(ale.getScreen(), ale.getRAM(), F);
		updateQValues(F, Q);
		currentAction = epsilonGreedy(Q);
		//Repeat(for each step of episode) until game is over:
		gettimeofday(&tvBegin, NULL);
		//This also stops when the maximum number of steps per episode is reached
		while(!ale.game_over()){
			reward.clear();
			reward.push_back(0.0);
			reward.push_back(0.0);
			updateQValues(F, Q);

			sanityCheck();
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

			delta = reward[0] + gamma * Qnext[nextAction] - Q[currentAction];

			updateReplTrace(currentAction, F);
			//Update weights vector:
			for(unsigned int a = 0; a < e.size(); a++){
				for(auto it=e[a].begin();it!=e[a].end();it++){
					long long idx = it->first;
                    float changeAmount = (alpha/maxFeatVectorNorm)*delta*it->second;
                    if (changeAmount!=0){
                        auto got = w[a].find(idx);
                        if (got==w[a].end()){
                            w[a][idx]=changeAmount;
                        }else{
                            got->second+=changeAmount;
                        }
                    }
				}
			}
			F = Fnext;
			currentAction = nextAction;
		}
		gettimeofday(&tvEnd, NULL);
		timeval_subtract(&tvDiff, &tvEnd, &tvBegin);
		elapsedTime = double(tvDiff.tv_sec) + double(tvDiff.tv_usec)/1000000.0;
		
		double fps = double(ale.getEpisodeFrameNumber())/elapsedTime;
		printf("episode: %d,\t%.0f points,\tavg. return: %.1f,\t%d frames,\t%.0f fps\n",
			episode, cumReward - prevCumReward, (double)cumReward/(episode),
			ale.getEpisodeFrameNumber(), fps);
        episodeResults.push_back(cumReward-prevCumReward);
        episodeFrames.push_back(ale.getEpisodeFrameNumber());
        episodeFps.push_back(fps);
		totalNumberFrames += ale.getEpisodeFrameNumber();
		prevCumReward = cumReward;
		ale.reset_game();
		if(toSaveCheckPoint && totalNumberFrames>saveThreshold){
            saveCheckPoint(episode,totalNumberFrames,episodeResults,saveWeightsEveryXFrames,episodeFrames,episodeFps);
            saveThreshold+=saveWeightsEveryXFrames;
        }
	}
}

void SarsaLearner::evaluatePolicy(ALEInterface& ale, Features *features){
	double reward = 0;
	double cumReward = 0; 
	double prevCumReward = 0;
	struct timeval tvBegin, tvEnd, tvDiff;
	double elapsedTime;

    std::string oldName = checkPointName+"-Result-writing.txt";
    std::string newName = checkPointName+"-Result-finished.txt";
    std::ofstream resultFile;
    resultFile.open(oldName.c_str());
    
	//Repeat (for each episode):
	for(int episode = 1; episode < numEpisodesEval; episode++){
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
		gettimeofday(&tvEnd, NULL);
		timeval_subtract(&tvDiff, &tvEnd, &tvBegin);
		elapsedTime = double(tvDiff.tv_sec) + double(tvDiff.tv_usec)/1000000.0;
		double fps = double(ale.getEpisodeFrameNumber())/elapsedTime;

        resultFile<<"Episode "<<episode<<": "<<cumReward-prevCumReward<<std::endl;
		printf("episode: %d,\t%.0f points,\tavg. return: %.1f,\t%d frames,\t%.0f fps\n", 
			episode, (cumReward-prevCumReward), (double)cumReward/(episode), ale.getEpisodeFrameNumber(), fps);

		ale.reset_game();
		prevCumReward = cumReward;
	}
    resultFile<<"Average: "<<(double)cumReward/numEpisodesEval<<std::endl;
    resultFile.close();
    rename(oldName.c_str(),newName.c_str());
    
}

void SarsaLearner::saveWeightsToFile(string suffix){
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

void SarsaLearner::loadWeights(){
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
