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
	maxFeatVectorNorm = 1;
	actionBeingPlayed = -1;
	pathToSaveLearnedWeights = param->outputPath;

	actions = ale.getLegalActionSet();

	numOptions      = param->numOptions;
	numBasicActions = actions.size();
	numTotalActions = numBasicActions + numOptions;

	numFeatures = bproFeatures.getNumberOfFeatures();
	
	for(int i = 0; i < numTotalActions; i++){
		//Initialize Q;
		Q.push_back(0);
		Qnext.push_back(0);
		//Initialize e:
		e.push_back(vector<float>(numFeatures, 0.0));
		w.push_back(vector<float>(numFeatures, 0.0));
		nonZeroElig.push_back(vector<int>());
	}
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
		updateQValues(F, Q);
		currentAction = epsilonGreedy(Q);
		//Repeat(for each step of episode) until game is over:
		gettimeofday(&tvBegin, NULL);

		//This also stops when the maximum number of steps per episode is reached
		while(!ale.game_over()){
			updateQValues(F, Q);

			sanityCheck();
			//Take action, observe reward and next state:
			r_alg = 0.0, r_real = 0.0;
			actionBeingPlayed = currentAction;
			act(ale, currentAction, learnedOptions);
			cumReward  += r_real;
			if(!ale.game_over()){
				//Obtain active features in the new state:
				Fnext.clear();
				bproFeatures.getActiveFeaturesIndices(ale.getScreen(), Fnext);
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

			delta = r_alg + GAMMA * Qnext[nextAction] - Q[currentAction];
			updateReplTrace(currentAction, F);

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

	if(action < numBasicActions){
		r_real += ale.act(actions[action]);
	} 
	else{
		int option_idx = action - numBasicActions;
		playOption(ale, option_idx, learnedOptions);
	}

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

void Learner::playOption(ALEInterface& ale, int option, vector<vector<vector<float> > > &learnedOptions){

	int currentAction;
	vector<int> Fbpro;
	float termProb = 0.0;
	int numActionsInOption = learnedOptions[option].size();
	
	vector<float> Q(numActionsInOption, 0.0);

	/* Rationale: Right now we have 'hierarchical' options. Options in the
	   lower level can act over the 18 actions, and then they should take
	   shorter than options that can call other options. Because the frame
	   skip is 5, what we do is the following: the low-level options have
	   a prob. of 0.05 of finishing, which corresponds on expectation to 20
	   steps (x5 ~ 1.6s). High level options have a termination prob. of 0.2
	   because if they call a low-level option they will have a termination
	   probability of 0.2 * 0.05 in fact (100 steps x 5 ~ 8.3s). */
	if(numActionsInOption > actions.size()){
		termProb = PROB_TERMINATION_L2;
	}
	else{
		termProb = PROB_TERMINATION_L1;
	}

	while(rand() % 1000 > 1000 * termProb && !ale.game_over()){
		//Get state and features active on that state:		
		Fbpro.clear();
		bproFeatures.getActiveFeaturesIndices(ale.getScreen(), Fbpro);

		//Update Q-values for each possible action
		for(int a = 0; a < numActionsInOption; a++){
			float sumW = 0;
			for(unsigned int i = 0; i < Fbpro.size(); i++){
				sumW += learnedOptions[option][a][Fbpro[i]];
			}
			Q[a] = sumW;
		}

		currentAction = epsilonGreedy(Q);

		if(toInterruptOption(ale, actionBeingPlayed, Fbpro, learnedOptions)){
			return;
		}
		/* Now things get nasty. We need to do it recursively because one 
		option can call another one. Hopefully everything is going to work.*/
		this->act(ale, currentAction, learnedOptions);
	}
}

bool Learner::toInterruptOption(ALEInterface& ale, int currentOption, vector<int> &Features, 
	vector<vector<vector<float> > > &learnedOptions){

	vector<float> Q(w.size(), 0.0);    //Q(a) entries
	//Update Q-values for each possible action
	for(int a = 0; a < w.size(); a++){
		float sumW = 0;
		for(unsigned int i = 0; i < Features.size(); i++){
			sumW += w[a][Features[i]];
		}
		Q[a] = sumW;
	}

	int bestAction = Mathematics::argmax(Q);

	if(Q[bestAction] - Q[currentOption] < 10e-3){
		return false;
	} else {
		//this->act(ale, bestAction, learnedOptions);
		return true;
	}
}

int Learner::epsilonGreedy(vector<float> &QValues){

	int action = Mathematics::argmax(QValues);
	//With probability epsilon: a <- random action in A(s)
	int random = rand();
	if(random % 1000 < EPSILON * 1000){
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

void Learner::updateReplTrace(int action, vector<int> &Features){
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

	//For all i in Fa:
	for(unsigned int i = 0; i < Features.size(); i++){
		int idx = Features[i];
		//If the trace is zero it is not in the vector
		//of non-zeros, thus it needs to be added
		if(e[action][idx] == 0){
	       nonZeroElig[action].push_back(idx);
	    }
		e[action][idx] = 1;
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
