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
			updateQValues(Fbpro, Q, option); //Update Q-values for each possible action
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

void Agent::updateQValues(vector<int> &Features, vector<float> &QValues, int option){
	for(int a = 0; a < numberOfAvailActions; a++){
		float sumW = 0;
		for(unsigned int i = 0; i < Features.size(); i++){
			sumW += w[option][a][Features[i]];
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

