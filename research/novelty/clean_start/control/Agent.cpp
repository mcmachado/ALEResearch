/* Author: Marlos C. Machado */

#include "Agent.hpp"
#include "../common/Mathematics.hpp"

#define NUM_BITS 1024

Agent::Agent(ALEInterface& ale, Parameters *param) : bproFeatures(param) {
	//Get the number of effective actions:
	if(param->isMinimalAction){
		actions = ale.getMinimalActionSet();
	}
	else{
		actions = ale.getLegalActionSet();
	}

	numberOfOptions          = 0;
	numberOfPrimitiveActions = actions.size();
	numberOfAvailActions     = numberOfPrimitiveActions + numberOfOptions;

	for(int i = 0; i < 2 * NUM_BITS; i++){
		freqOfBitFlips.push_back(0.0);
		transitions.push_back(0.0);
	}

	for(int i = 0; i < numberOfOptions; i++){
		w.push_back(vector< vector<float> >(numberOfPrimitiveActions, vector<float>(bproFeatures.getNumberOfFeatures(), 0.0)));
	}
}

void Agent::updateAverage(Parameters *param, vector<bool> Fprev, vector<bool> F, 
	int frame, int iter, vector<vector<bool> > &dataset){

	assert (Fprev.size() == F.size());
	assert (F.size() == NUM_BITS);
	
	bool toStore = false;
	/*This should be a parameter, but in fact I do not plan to use it as true, so I did not set it
	  in the class Parameters. Eventually, if I ever want to use it, I have to change it here. */
	bool toReportAll_param = false;
	/*This is used to save intermediate processing steps, like the rare events. It is not defined
	  in the Parameters class because I hope this is used only internally. */
	std::stringstream sstm_fileName;
	sstm_fileName << "frequencyRareEventsIter" << iter + 1 << "_bits.csv";
	string outputPath_param = sstm_fileName.str();

	vector<int> tempVector(2 * NUM_BITS, 0);

	for(int i = 0; i < NUM_BITS; i++){
		if(!Fprev[i] && F[i]){ // 0->1
			freqOfBitFlips[i] = (freqOfBitFlips[i] * (frame - 1) + 1) / frame;
			tempVector[i] = 1;
			if(frame > param->framesToDefRarity && freqOfBitFlips[i] < param->rarityFreqThreshold){
				tempVector[i] = 2; //1 denotes flip, 2 denotes relevant flip
				toStore = true;
			}
		} else{
			freqOfBitFlips[i] = (freqOfBitFlips[i] * (frame - 1) + 0) / frame;
		}		
		if(Fprev[i] && !F[i]){ // 1->0
			freqOfBitFlips[i + NUM_BITS] = (freqOfBitFlips[i + NUM_BITS] * (frame - 1) + 1) / frame;
			tempVector[i + NUM_BITS] = 1;
			if(frame > param->framesToDefRarity && freqOfBitFlips[i + NUM_BITS] < param->rarityFreqThreshold){
				tempVector[i + NUM_BITS] = 2;
				toStore = true;
			}
		} else{
			freqOfBitFlips[i + NUM_BITS] = (freqOfBitFlips[i + NUM_BITS] * (frame - 1) + 0) / frame;
		}
	}

	ofstream myFileBits;
	myFileBits.open (outputPath_param, ios::app);
	vector<int> bytesToPrint;
	if(toStore){
		for(int i = 0; i < tempVector.size(); i++){
			if(toReportAll_param == 1){
				if(tempVector[i] != 0){
					myFileBits << i << ",";
					dataset[i].push_back(true);
				}
				else{
					dataset[i].push_back(false);
				}
			}
			else{
				if(tempVector[i] == 2){
					myFileBits << i << ",";
					dataset[i].push_back(true);
				}
				else{
					dataset[i].push_back(false);
				}
			}
			/* I USED THIS CODE BEFORE FILLING dataset:
			if(toReportAll_param == 1 && tempVector[i] != 0){
				myFileBits << i << ",";

			}
			if(toReportAll_param == 0 && tempVector[i] == 2){
				myFileBits << i << ",";
			}
			*/			
		}
		myFileBits << endl;
	}
	myFileBits.close();	
}

int Agent::playActionUpdatingAvg(ALEInterface& ale, Parameters *param, int &frame, 
	int nextAction, int iter, vector<vector<bool> > &dataset){

	vector<bool> F(NUM_BITS, 0); //Set of active features
	vector<bool> Fprev;
	int reward = 0;

	//If the selected action was one of the primitive actions
	if(nextAction < numberOfPrimitiveActions){
		F.clear();
		ramFeatures.getCompleteFeatureVector(ale.getRAM(), F);
		F.pop_back();
		for(int i = 0; i < param->numStepsPerAction; i++){
			reward += ale.act((Action) nextAction);
			frame++;
			Fprev.swap(F);
			F.clear();
			ramFeatures.getCompleteFeatureVector(ale.getRAM(), F);
			F.pop_back();
			updateAverage(param, Fprev, F, frame, iter, dataset);
		}
	}
	//If the selected action was one of the options
	else{
		int currentAction;
		vector<int> Fbpro;	//Set of features active
		vector<float> Q(numberOfPrimitiveActions, 0.0);    //Q(a) entries

		int option = nextAction - numberOfPrimitiveActions;
		while(rand()%1000 > 1000 * param->optionTerminationProb && !ale.game_over()){
			F.clear();
			ramFeatures.getCompleteFeatureVector(ale.getRAM(), F);
			F.pop_back();
			//Get state and features active on that state:		
			Fbpro.clear();
			bproFeatures.getActiveFeaturesIndices(ale.getScreen(), Fbpro);
			updateQValues(learnedOptions[option], Fbpro, Q, option); //Update Q-values for each possible action
			currentAction = epsilonGreedy(Q, param->epsilon);
			//Take action, observe reward and next state:
			reward += ale.act((Action) currentAction);
			frame++;
			Fprev.swap(F);
			F.clear();
			ramFeatures.getCompleteFeatureVector(ale.getRAM(), F);
			F.pop_back();
			updateAverage(param, Fprev, F, frame, iter, dataset);
		}
	}
	return reward;
}

/**
 * The first parameter is the one that is used by Sarsa. The second is used to
 * pass aditional information to the running algorithm (like 'real score' if one
 * is using a surrogate reward function).
 */
void Agent::act(ALEInterface& ale, int action, Parameters *param,
	std::vector<float> &mean, std::vector<float> &std,
	std::vector<float> &eigenVectors, vector<float> &reward){

	vector<bool> FRam, FnextRam;
	float r_alg = 0.0, r_real = 0.0;

	FRam.clear();
	ramFeatures.getCompleteFeatureVector(ale.getRAM(), FRam);

	if(action < numberOfPrimitiveActions){
		r_real = ale.act(actions[action]);
	}
	else{
		int option_idx = action - numberOfPrimitiveActions;
		r_real = playOption(ale, param->epsilon, option_idx);
	}

	FnextRam.clear();
	ramFeatures.getCompleteFeatureVector(ale.getRAM(), FnextRam);
	updateTransitionVector(FRam, FnextRam);

	for(int i = 0; i < transitions.size(); i++){
		transitions[i] = (transitions[i] - mean[i])/std[i];
		r_alg += eigenVectors[i] * transitions[i];
	}

	reward[0] = r_alg;
	reward[1] = r_real;
}


int Agent::playOption(ALEInterface& ale, float epsilon, int option){

	int r_real = 0;
	float termProb = 0.01;
	int currentAction;
	vector<int> Fbpro;	                      //Set of features active
	vector<float> Q(numberOfPrimitiveActions, 0.0);    //Q(a) entries

	RAMFeatures ramFeatures;
	vector<bool> F(NUM_BITS, 0); //Set of active features
	vector<bool> Fprev;

	while(rand()%1000 > 1000 * termProb && !ale.game_over()){
		//Get state and features active on that state:
		Fbpro.clear();
		bproFeatures.getActiveFeaturesIndices(ale.getScreen(), Fbpro);

		//Update Q-values for each possible action
		for(int a = 0; a < numberOfPrimitiveActions; a++){
			float sumW = 0;
			for(unsigned int i = 0; i < Fbpro.size(); i++){
				sumW += learnedOptions[option][a][Fbpro[i]];
			}
			Q[a] = sumW;
		}

		currentAction = epsilonGreedy(Q, epsilon);
		//Take action, observe reward and next state:
		r_real += ale.act((Action) currentAction);
		Fprev.swap(F);
		F.clear();
		ramFeatures.getCompleteFeatureVector(ale.getRAM(), F);
		F.pop_back();
	}
	return r_real;
}

void Agent::updateTransitionVector(vector<bool> F, vector<bool> Fnext){
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

void Agent::updateQValues(vector<vector<float> > &weights, vector<int> &Features, vector<float> &QValues, int option){
	for(int a = 0; a < numberOfAvailActions; a++){
		float sumW = 0;
		for(unsigned int i = 0; i < Features.size(); i++){
			sumW += weights[a][Features[i]];
		}
		QValues[a] = sumW;
	}
}

int Agent::epsilonGreedy(vector<float> &QValues, float epsilon){
	int randomActionTaken = 0;

	int action = Mathematics::argmax(QValues);
	//With probability epsilon: a <- random action in A(s)
	int random = rand();
	if(int(nearbyint(float(rand()%1000 < epsilon * 1000))) == 0){
		randomActionTaken = 1;
		action = rand() % numberOfPrimitiveActions;
	}
	return action;
}

void Agent::updateReplTrace(Parameters *param, int action, vector<int> &Features){
	//e <- gamma * lambda * e
	for(unsigned int a = 0; a < nonZeroElig.size(); a++){
		int numNonZero = 0;
	 	for(unsigned int i = 0; i < nonZeroElig[a].size(); i++){
	 		int idx = nonZeroElig[a][i];
	 		//To keep the trace sparse, if it is
	 		//less than a threshold it is zero-ed.
			e[a][idx] = param->gamma * param->lambda * e[a][idx];
			if(e[a][idx] < param->traceThreshold){
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

void Agent::cleanTraces(){
	for(unsigned int a = 0; a < nonZeroElig.size(); a++){
		for(unsigned int i = 0; i < nonZeroElig[a].size(); i++){
			int idx = nonZeroElig[a][i];
			e[a][idx] = 0.0;
		}
		nonZeroElig[a].clear();
	}
}

void Agent::sanityCheck(vector<float> &QValues){
	for(int i = 0; i < numberOfAvailActions; i++){
		if(fabs(QValues[i]) > 10e7 || QValues[i] != QValues[i] /*NaN*/){
			printf("It seems your algorithm diverged!\n");
			exit(0);
		}
	}
}