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
#include "SarsaSplitLearner.hpp"

#include <stdio.h>
#include <math.h>

using namespace std;

//Freeway: Chicken height: 0x8E
//Private Eye: Screen number: 0xDC

SarsaSplitLearner::SarsaSplitLearner(Environment<bool>& env, Parameters *param, int seed) : RLLearner<bool>(env, param, seed) {
    totalNumberFrames = 0.0;
    maxFeatVectorNorm = 1;
	
	deltaW = 0.0;
	deltaPsi = 0.0;
	alphaW = param->getAlpha();
	lambdaW = param->getLambda();
	alphaPsi = param->getAuxAlpha();
	lambdaPsi = param->getAuxLambda();
	gammaPsi = param->getAuxGamma();
	traceThreshold = param->getTraceThreshold();
	numFeatures = env.getNumberOfFeatures();
	toSaveCheckPoint = param->getToSaveCheckPoint();
	saveWeightsEveryXSteps = param->getFrequencySavingWeights();
	pathWeightsFileToLoad = param->getPathToWeightsFiles();
	featureSeen.resize(numActions);
	numEpisodesEval = param->getNumEpisodesEval();
	numEpisodesLearn = param->getNumEpisodesLearn();
	episodeLength = param->getEpisodeLength();

	for(int i = 0; i < numActions; i++){
		//Initialize Q;
		Q.push_back(0);
		QW.push_back(0);
		QnextW.push_back(0);
		QPsi.push_back(0);
		QnextPsi.push_back(0);
		//Initialize e:
		Fcount.push_back(vector<int>(numFeatures, 0));
		eW.push_back(vector<float>(numFeatures, 0.0));
		ePsi.push_back(vector<float>(numFeatures, 0.0));
		w.push_back(vector<float>(numFeatures, 0.0));
		psi.push_back(vector<float>(numFeatures, param->getDegreeOfOptimism()));
		nonZeroEligW.push_back(vector<int>());
		nonZeroEligPsi.push_back(vector<int>());
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

SarsaSplitLearner::~SarsaSplitLearner(){}

void SarsaSplitLearner::updateReplTrace(int action, vector<int> &Features){
	//e <- gamma * lambda * e
	for(unsigned int a = 0; a < nonZeroEligW.size(); a++){
		int numNonZero = 0;
	 	for(unsigned int i = 0; i < nonZeroEligW[a].size(); i++){
	 		int idx = nonZeroEligW[a][i];
	 		//To keep the trace sparse, if it is
	 		//less than a threshold it is zero-ed.
			eW[a][idx] = gamma * lambdaW * eW[a][idx];
			if(eW[a][idx] < traceThreshold){
				eW[a][idx] = 0;
			}
			else{
				nonZeroEligW[a][numNonZero] = idx;
		  		numNonZero++;
			}
		}
		nonZeroEligW[a].resize(numNonZero);
	}
	//For all i in Fa:
	for(unsigned int i = 0; i < Features.size(); i++){
		int idx = Features[i];
		//If the trace is zero it is not in the vector
		//of non-zeros, thus it needs to be added
		if(eW[action][idx] == 0){
	       nonZeroEligW[action].push_back(idx);
	    }
		eW[action][idx] = 1;
	}

	//e <- gamma * lambda * e
	for(unsigned int a = 0; a < nonZeroEligPsi.size(); a++){
		int numNonZero = 0;
	 	for(unsigned int i = 0; i < nonZeroEligPsi[a].size(); i++){
	 		int idx = nonZeroEligPsi[a][i];
	 		//To keep the trace sparse, if it is
	 		//less than a threshold it is zero-ed.
			ePsi[a][idx] = gammaPsi * lambdaPsi * ePsi[a][idx];
			if(ePsi[a][idx] < traceThreshold){
				ePsi[a][idx] = 0;
			}
			else{
				nonZeroEligPsi[a][numNonZero] = idx;
		  		numNonZero++;
			}
		}
		nonZeroEligPsi[a].resize(numNonZero);
	}

	//For all i in Fa:
	for(unsigned int i = 0; i < Features.size(); i++){
		int idx = Features[i];
		//If the trace is zero it is not in the vector
		//of non-zeros, thus it needs to be added
		if(ePsi[action][idx] == 0){
	       nonZeroEligPsi[action].push_back(idx);
	    }
		ePsi[action][idx] = 1;
	}
}

/*
void SarsaSplitLearner::updateAcumTrace(int action, vector<int> &Features){
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

	//For all i in Fa:
	for(unsigned int i = 0; i < Features.size(); i++){
		int idx = Features[i];
		//If the trace is zero it is not in the vector
		//of non-zeros, thus it needs to be added
		if(e[action][idx] == 0){
	       nonZeroElig[action].push_back(idx);
	    }
		e[action][idx] += 1;
	}
}
*/

void SarsaSplitLearner::sanityCheck(std::vector<float> &QValues){
	for(int i = 0; i < numActions; i++){
		if(fabs(QValues[i]) > 10e7 || QValues[i] != QValues[i] /*NaN*/){
			printf("It seems your algorithm diverged!\n");
			exit(0);
		}
	}
}

//To do: we do not want to save weights that are zero
void SarsaSplitLearner::saveCheckPoint(int episode, int totalNumberFrames, vector<float>& episodeResults,int& frequency,vector<int>& episodeFrames, vector<double>& episodeFps){
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
    for (int a=0;a<featureSeen.size();a++){
        for (int index=0; index<featureSeen[a].size();index++){
            checkPointFile<<a<<" "<<featureSeen[a][index]<<" "<<w[a][featureSeen[a][index]]<<"\t";
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

void SarsaSplitLearner::loadCheckPoint(ifstream& checkPointToLoad){
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

void SarsaSplitLearner::learnPolicy(Environment<bool>& env){
	
	struct timeval tvBegin, tvEnd, tvDiff;
	vector<float> reward;
	float elapsedTime;
	float cumReward = 0, prevCumReward = 0;
	sawFirstReward = 0; firstReward = 1.0;
	vector<float> episodeResults;
    vector<int> episodeFrames;
    vector<double> episodeFps;

	//Repeat (for each episode):
	for(int episode = episodePassed + 1; episode <= numEpisodesLearn; episode++){
		double disc_return = 0.0;
		//We have to clean the traces every episode:
		for(unsigned int a = 0; a < nonZeroEligW.size(); a++){
			for(unsigned int i = 0; i < nonZeroEligW[a].size(); i++){
				int idx = nonZeroEligW[a][i];
				eW[a][idx] = 0.0;
			}
			nonZeroEligW[a].clear();
		}
		for(unsigned int a = 0; a < nonZeroEligPsi.size(); a++){
			for(unsigned int i = 0; i < nonZeroEligPsi[a].size(); i++){
				int idx = nonZeroEligPsi[a][i];
				ePsi[a][idx] = 0.0;
			}
			nonZeroEligPsi[a].clear();
		}

		F.clear();
		env.getActiveFeaturesIndices(F);
		updateQValues(F, w, QW); updateQValues(F, psi, QPsi);
		for(int i = 0; i < Q.size(); i++){
			Q[i] = QW[i] + QPsi[i];
		}

		currentAction = epsilonGreedy(Q);

		//Repeat(for each step of episode) until game is over:
		double t = 0.0;
		gettimeofday(&tvBegin, NULL);
		//This also stops when the maximum number of steps per episode is reached
		while(!env.isTerminal() && episodeLength > env.getEpisodeFrameNumber()){
			t += 1.0;
			reward.clear();
			reward.push_back(0.0);
			reward.push_back(0.0);
			updateQValues(F, w, QW); updateQValues(F, psi, QPsi);
			sanityCheck(QW); sanityCheck(QPsi);

			for(int i = 0; i < F.size(); i++){
				Fcount[currentAction][F[i]]++;
			}

			//Take action, observe reward and next state:
			act(env, currentAction, reward);
			cumReward  += reward[1];
			disc_return = reward[1] + gamma * disc_return;

			if(!env.isTerminal()){
				//Obtain active features in the new state:
				Fnext.clear();
				env.getActiveFeaturesIndices(Fnext);

				updateQValues(Fnext, w, QnextW); updateQValues(Fnext, psi, QnextPsi);
				for(int i = 0; i < Q.size(); i++){
					Q[i] = QnextW[i] + QnextPsi[i];
				}
				nextAction = epsilonGreedy(Q);
			}
			else{
				nextAction = 0;
				for(unsigned int i = 0; i < QnextW.size(); i++){
					QnextW[i] = 0;
					QnextPsi[i] = 0;
				}
			}
			//To ensure the learning rate will never increase along
			//the time, Marc used such approach in his JAIR paper		
			if (F.size() > maxFeatVectorNorm){
				maxFeatVectorNorm = F.size();
			}

			deltaW = reward[0] + gamma * QnextW[nextAction] - QW[currentAction];
			deltaPsi = gammaPsi * QnextPsi[nextAction] - QPsi[currentAction];

			updateReplTrace(currentAction, F);
			//Update weights vector:
			for(unsigned int a = 0; a < nonZeroEligW.size(); a++){
				for(unsigned int i = 0; i < nonZeroEligW[a].size(); i++){
					int idx = nonZeroEligW[a][i];
					if (w[a][idx] == 0 && deltaW != 0){
                        featureSeen[a].push_back(idx);
                    }
                    float stepSizeW = alphaW/maxFeatVectorNorm;
                    //float stepSizeW = (alphaW/maxFeatVectorNorm) * (1.0/(float)Fcount[a][idx]);
					w[a][idx] = w[a][idx] + stepSizeW * deltaW * eW[a][idx];
				}
			}

			for(unsigned int a = 0; a < nonZeroEligPsi.size(); a++){
				for(unsigned int i = 0; i < nonZeroEligPsi[a].size(); i++){
					int idx = nonZeroEligPsi[a][i];
					//float stepSizeExpPsi = alphaPsi/maxFeatVectorNorm;
					//float stepSizeExpPsi = (alphaPsi/maxFeatVectorNorm) * (1.0/(float)env.getEpisodeFrameNumber()/*Fcount[a][idx]*/);
					float stepSizeExpPsi = (alphaPsi/maxFeatVectorNorm) * (1.0 - sqrt(float(/*Fcount[a][idx]*/env.getEpisodeFrameNumber()-1)/(float)/*Fcount[a][idx]*/env.getEpisodeFrameNumber()));
//					printf("psi[%d] = %f + %f * %f * %f\n", idx, psi[a][idx], stepSizeExpPsi, deltaPsi, ePsi[a][idx]);
					psi[a][idx] = psi[a][idx] + stepSizeExpPsi * deltaPsi * ePsi[a][idx];
				}
			}
			F = Fnext;
			currentAction = nextAction;

			printf("%f %f %f %f\n", psi[0][0], psi[0][1], psi[0][2], psi[0][3]);
		}
		gettimeofday(&tvEnd, NULL);
		timeval_subtract(&tvDiff, &tvEnd, &tvBegin);
		elapsedTime = float(tvDiff.tv_sec) + float(tvDiff.tv_usec)/1000000.0;
		
		float fps = float(env.getEpisodeFrameNumber())/elapsedTime;
		//printf("episode: %d,\t disc. return: %f,\t cum. return: %f,\t reward/step: %f\n", episode, cumReward, (cumReward - prevCumReward), (cumReward - prevCumReward)/(float)env.getEpisodeFrameNumber());
        episodeResults.push_back(cumReward-prevCumReward);
        episodeFrames.push_back(env.getEpisodeFrameNumber());
        episodeFps.push_back(fps);
		totalNumberFrames += env.getEpisodeFrameNumber();
		prevCumReward = cumReward;
		env.reset_game();
		if(toSaveCheckPoint && episode%saveWeightsEveryXSteps == 0){
            saveCheckPoint(episode,totalNumberFrames,episodeResults,saveWeightsEveryXSteps,episodeFrames,episodeFps);
        }
	}
}

double SarsaSplitLearner::evaluatePolicy(Environment<bool>& env){
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
	for(episode = 1; episode < numEpisodesEval; episode++){
		//Repeat(for each step of episode) until game is over:
		for(int step = 0; !env.isTerminal() && step < episodeLength; step++){
			//Get state and features active on that state:		
			F.clear();
			env.getActiveFeaturesIndices(F);
			updateQValues(F, w, QW);       //Update Q-values for each possible action
			currentAction = epsilonGreedy(QW);
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

void SarsaSplitLearner::saveWeightsToFile(string suffix){
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

void SarsaSplitLearner::loadWeights(){
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
