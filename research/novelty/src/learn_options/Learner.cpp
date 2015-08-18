#include "Learner.hpp"
#include "constants.hpp"
#include "../common/Timer.hpp"
#include "../common/Mathematics.hpp"

using namespace std;

Learner::Learner(ALEInterface& ale, Parameters *param) : bproFeatures(param->gameName){
	delta = 0.0;
	cumReward = 0; 
	prevCumReward = 0;
	cumIntrReward = 0;
	prevCumIntrReward = 0;
	maxFeatVectorNorm = 1;

	for(int i = 0; i < (ramFeatures.getNumberOfFeatures() - 1)*2; i++){
		transitions.push_back(0);
	}

	actions = ale.getLegalActionSet();

	numOptions      = param->numOptions;
	numBasicActions = actions.size();
	numTotalActions = numBasicActions + numOptions;

	//Reading file containing the vector that describes the reward for the option learning
	//The first X positions encode the transition 0->1 and the other X encode 1->0.
	pathToRewardDescription = param->eigVectorPath;
	std::ifstream infile1(pathToRewardDescription.c_str());
	float value;
	while(infile1 >> value){
		eigVector.push_back(value);
	}
	pathToStatsDescription = param->statEigVectorPath;
	std::ifstream infile2((pathToStatsDescription + "_mean.out").c_str());
	while(infile2 >> value){
		mean.push_back(value);
	}
	std::ifstream infile3((pathToStatsDescription + "_std.out").c_str());
	while(infile3 >> value){
		std.push_back(value);
	}

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
	
	vector<float> reward;
	//Repeat (for each episode):
	int episode, totalNumberFrames = 0;
	//This is going to be interrupted by the ALE code since I set max_num_frames beforehand
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
			reward.clear();
			reward.push_back(0.0);
			reward.push_back(0.0);
			updateQValues(F, Q);			

			sanityCheck();
			//Take action, observe reward and next state:
			act(ale, currentAction, reward, learnedOptions);
			cumIntrReward += reward[0];
			cumReward  += reward[1];
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

			delta = reward[0] + GAMMA * Qnext[nextAction] - Q[currentAction];

			updateReplTrace(currentAction, F);
			//Update weights vector:
			for(unsigned int a = 0; a < nonZeroElig.size(); a++){
				for(unsigned int i = 0; i < nonZeroElig[a].size(); i++){
					int idx = nonZeroElig[a][i];
					w[a][idx] = w[a][idx] + (ALPHA/maxFeatVectorNorm) * delta * e[a][idx];
				}
			}
			F = Fnext;
			FRam = FnextRam;
			currentAction = nextAction;
		}
		gettimeofday(&tvEnd, NULL);
		timeval_subtract(&tvDiff, &tvEnd, &tvBegin);
		elapsedTime = float(tvDiff.tv_sec) + float(tvDiff.tv_usec)/1000000.0;
		
		float fps = float(ale.getEpisodeFrameNumber())/elapsedTime;
		printf("episode: %d,\t%.0f points,\tavg. return: %.1f,\tnovelty reward: %.2f (%.2f),\t%d frames,\t%.0f fps\n",
			episode + 1, cumReward - prevCumReward, (float)cumReward/(episode + 1.0),
			cumIntrReward - prevCumIntrReward, cumIntrReward/(episode + 1.0), ale.getEpisodeFrameNumber(), fps);
		totalNumberFrames += ale.getEpisodeFrameNumber();
		prevCumReward = cumReward;
		prevCumIntrReward = cumIntrReward;
		ale.reset_game();
	}
	
	stringstream ss;
	ss << episode;
	saveWeightsToFile(ss.str());
}

void Learner::act(ALEInterface& ale, int action, vector<float> &reward, vector<vector<vector<float> > > &learnedOptions){

	float r_alg = 0.0, r_real = 0.0;

	FRam.clear();
	ramFeatures.getCompleteFeatureVector(ale.getRAM(), FRam);

	if(action < numBasicActions){
		r_real = ale.act(actions[action]);
	} 
	else{
		int option_idx = action - numBasicActions;
		//r_real = playOption(ale, option_idx, learnedOptions);
	}

	FnextRam.clear();
	ramFeatures.getCompleteFeatureVector(ale.getRAM(), FnextRam);
	updateTransitionVector(FRam, FnextRam);

	for(int i = 0; i < transitions.size(); i++){
		transitions[i] = (transitions[i] - mean[i])/std[i];
		r_alg += eigVector[i] * transitions[i];
	}

	reward[0] = r_alg;
	reward[1] = r_real;
}

int Learner::playOption(ALEInterface& ale, int option, vector<vector<vector<float> > > &learnedOptions){

	int r_real = 0;
	float termProb = 0.01;
	int currentAction;
	vector<int> Fbpro;	                      //Set of features active
	vector<float> Q(numBasicActions, 0.0);    //Q(a) entries

	vector<bool> F(NUM_BITS, 0); //Set of active features
	vector<bool> Fprev;

	while(rand()%1000 > 1000 * termProb && !ale.game_over()){
		//Get state and features active on that state:		
		Fbpro.clear();
		bproFeatures.getActiveFeaturesIndices(ale.getScreen(), Fbpro);

		//Update Q-values for each possible action
		for(int a = 0; a < numBasicActions; a++){
			float sumW = 0;
			for(unsigned int i = 0; i < Fbpro.size(); i++){
				sumW += learnedOptions[option][a][Fbpro[i]];
			}
			Q[a] = sumW;
		}

		currentAction = epsilonGreedy(Q);
		//Take action, observe reward and next state:
		r_real += ale.act((Action) currentAction);
		Fprev.swap(F);
		F.clear();
		ramFeatures.getCompleteFeatureVector(ale.getRAM(), F);
		F.pop_back();
	}
	return r_real;
}

void Learner::updateTransitionVector(vector<bool> F, vector<bool> Fnext){
	int numTransitionFeatures = F.size();
	
	for(int i = 0; i < F.size(); i++){
		if(!F[i] && Fnext[i]){ //0->1
			transitions[i] = 1;
		}
		else{
			transitions[i] = 0;	
		}
		if(F[i] && !Fnext[i]){ //1->0
			transitions[i + numTransitionFeatures - 1] = 1;
		}
		else{
			transitions[i + numTransitionFeatures - 1] = 0;	
		}
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
		if(fabs(Q[i]) > 10e7 || Q[i] != Q[i] /*NaN*/){
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