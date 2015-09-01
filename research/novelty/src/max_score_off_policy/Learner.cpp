#include "Learner.hpp"
#include "constants.hpp"
#include "../common/Timer.hpp"
#include "../common/Mathematics.hpp"

using namespace std;

Learner::Learner(ALEInterface& ale, Parameters *param) : bproFeatures(param->gameName){
	delta = 0.0;
	cumReward = 0;
	firstReward = 1.0;
	prevCumReward = 0;
	sawFirstReward = 0;
	randomActionTaken = 0;
	maxFeatVectorNorm = 1;
	pathToSaveLearnedWeights = param->outputPath;

	actions = ale.getLegalActionSet();

	numOptions      = param->numOptions;
	numBasicActions = actions.size();
	numTotalActions = numBasicActions + numOptions;

	numFeatures = bproFeatures.getNumberOfFeatures();
	
	for(int i = 0; i < numBasicActions; i++){
		//Initialize Q;
		Q.push_back(0);
		Qnext.push_back(0);
		//Initialize e:
		e.push_back(vector<float>(numFeatures, 0.0));
		w.push_back(vector<float>(numFeatures, 0.0));
		nonZeroElig.push_back(vector<int>());
	}
}

int Learner::actionFromOptions(vector<int> &Features, vector<vector<vector<float> > > &learnedOptions){
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

int Learner::getNextAction(vector<int> &Features, vector<float> &QValues, 
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

void Learner::learnPolicy(ALEInterface& ale, vector<vector<vector<float> > > &learnedOptions){
	
	//Repeat (for each episode):
	int episode, totalNumberFrames = 0;
	for(episode = 0; totalNumberFrames < MAX_NUM_FRAMES; episode++){ 
		//We have to clean the traces every episode:
		for(unsigned int a = 0; a < nonZeroElig.size(); a++){
			for(unsigned int i = 0; i < nonZeroElig[a].size(); i++){
				int idx = nonZeroElig[a][i];
				e[a][idx] = 0.0;
			}
			nonZeroElig[a].clear();
		}
		F.clear();
		bproFeatures.getActiveFeaturesIndices(ale.getScreen(), F);

		//Repeat(for each step of episode) until game is over:
		gettimeofday(&tvBegin, NULL);

		//This also stops when the maximum number of steps per episode is reached
		while(!ale.game_over()){
			updateQValues(F, Q);
			sanityCheck();
			//Take action, observe reward and next state:
			r_alg = 0.0, r_real = 0.0;
			currentAction = getNextAction(F, Q, episode, learnedOptions);
			act(ale, currentAction, learnedOptions);
			cumReward  += r_real;

			if(!ale.game_over()){
				//Obtain active features in the new state:
				Fnext.clear();
				bproFeatures.getActiveFeaturesIndices(ale.getScreen(), Fnext);
				updateQValues(Fnext, Qnext);     //Update Q-values for the new active features
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

			delta = r_alg + GAMMA * Qnext[nextAction] - Q[currentAction];

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
			float stepSize = ALPHA/maxFeatVectorNorm;
			for(unsigned int a = 0; a < nonZeroElig.size(); a++){
				for(unsigned int i = 0; i < nonZeroElig[a].size(); i++){
					int idx = nonZeroElig[a][i];
					w[a][idx] = w[a][idx] + stepSize * delta * e[a][idx];
				}
			}
			F = Fnext;
			currentAction = nextAction;
		}
		gettimeofday(&tvEnd, NULL);
		timeval_subtract(&tvDiff, &tvEnd, &tvBegin);
		elapsedTime = float(tvDiff.tv_sec) + float(tvDiff.tv_usec)/1000000.0;
		
		float fps = float(ale.getEpisodeFrameNumber())/elapsedTime;
		printf("episode: %d,\t%.0f points,\tavg. return: %.1f,\t%d frames,\t%.0f fps\n",
			episode + 1, cumReward - prevCumReward, (float)cumReward/(episode + 1.0),
			ale.getEpisodeFrameNumber(), fps);
		totalNumberFrames += ale.getEpisodeFrameNumber();
		prevCumReward = cumReward;
		ale.reset_game();
	}
	
	stringstream ss;
	ss << episode;
	saveWeightsToFile(ss.str());
}

void Learner::evaluatePolicy(ALEInterface& ale, vector<vector<vector<float> > > &learnedOptions){

}

void Learner::act(ALEInterface& ale, int action, vector<vector<vector<float> > > &learnedOptions){

	r_real += ale.act(actions[action]);

	/* Here I am letting the option return a single reward and then I am
	 normalizing over it. I don't know if it is not better to normalize
	 each step of the option. */
	if(r_real != 0.0){
		if(!sawFirstReward){
			firstReward = std::abs(r_real);
			sawFirstReward = 1;
		}
	}
	if(sawFirstReward){
		r_alg = r_real/firstReward;
	}
}

int Learner::epsilonGreedy(vector<float> &QValues){
	randomActionTaken = 0;

	int action = Mathematics::argmax(QValues);
	//With probability epsilon: a <- random action in A(s)
	int random = rand();
	if(random % 1000 < EPSILON * 1000){
		randomActionTaken = 1;
		action = rand() % QValues.size();
	}
	return action;
}

void Learner::sanityCheck(){
	for(int i = 0; i < numTotalActions; i++){
		if(fabs(Q[i]) > 10e6 || Q[i] != Q[i] /*NaN*/){
			printf("It seems your algorithm diverged!\n");
			exit(0);
		}
	}
}

void Learner::updateQValues(vector<int> &Features, vector<float> &QValues){
	for(int a = 0; a < numTotalActions; a++){
		float sumW = 0;
		for(unsigned int i = 0; i < Features.size(); i++){
			sumW += w[a][Features[i]];
		}
		QValues[a] = sumW;
	}
}

void Learner::updateReplTrace(int action){
	//e <- gamma * lambda * e
	for(unsigned int a = 0; a < nonZeroElig.size(); a++){
		int numNonZero = 0;
	 	for(unsigned int i = 0; i < nonZeroElig[a].size(); i++){
	 		int idx = nonZeroElig[a][i];
	 		//To keep the trace sparse, if it is
	 		//less than a threshold it is zero-ed.
			e[a][idx] = GAMMA * LAMBDA * e[a][idx];
			if(e[a][idx] < TRACE_THRESHOLD){
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

void Learner::saveWeightsToFile(string suffix){
	std::ofstream weightsFile ((pathToSaveLearnedWeights + suffix + ".wgt").c_str());
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
