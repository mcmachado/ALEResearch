/* Author: Marlos C. Machado */

#include "ControlAgent.hpp"
#include "../common/Mathematics.hpp"
#include "../observations/RAMFeatures.hpp"
#include "../observations/BPROFeatures.hpp"

#define NUM_BITS 1024

void updateAverage(Parameters *param, Agent &agent, vector<bool> Fprev, vector<bool> F, int frame, int iter){
	assert (Fprev.size() == F.size());
	assert (F.size() == NUM_BITS);
	
	bool toPrint = false;
	/*This should be a parameter, but in fact I do not plan to use it as true, so I did not set it
	  in the class Parameters. Eventually, if I ever want to use it, I have to change it here. */
	bool toReportAll_param = false;
	/*This is used to save intermediate processing steps, like the rare events. It is not defined
	  in the Parameters class because I hope this is used only internally. */
	std::stringstream sstm_fileName;
	sstm_fileName << "frequencyRareEventsIter" << iter << ".out";
	string outputPath_param = sstm_fileName.str();

	vector<int> tempVector(2 * NUM_BITS, 0);

	for(int i = 0; i < NUM_BITS; i++){
		if(!Fprev[i] && F[i]){ // 0->1
			agent.freqOfBitFlips[i] = (agent.freqOfBitFlips[i] * (frame - 1) + 1) / frame;
			tempVector[i] = 1;
			if(frame > param->framesToDefRarity && agent.freqOfBitFlips[i] < param->rarityFreqThreshold){
				tempVector[i] = 2; //1 denotes flip, 2 denotes relevant flip
				toPrint = true;
			}
		} else{
			agent.freqOfBitFlips[i] = (agent.freqOfBitFlips[i] * (frame - 1) + 0) / frame;
		}		
		if(Fprev[i] && !F[i]){ // 1->0
			agent.freqOfBitFlips[i + NUM_BITS] = (agent.freqOfBitFlips[i + NUM_BITS] * (frame - 1) + 1) / frame;
			tempVector[i + NUM_BITS] = 1;
			if(frame > param->framesToDefRarity && agent.freqOfBitFlips[i + NUM_BITS] < param->rarityFreqThreshold){
				tempVector[i + NUM_BITS] = 2;
				toPrint = true;
			}
		} else{
			agent.freqOfBitFlips[i + NUM_BITS] = (agent.freqOfBitFlips[i + NUM_BITS] * (frame - 1) + 0) / frame;
		}
	}

	ofstream myFileBits;
	myFileBits.open (outputPath_param + "_bits.csv", ios::app);
	vector<int> bytesToPrint;
	if(toPrint){
		for(int i = 0; i < tempVector.size(); i++){
			if(toReportAll_param == 1 && tempVector[i] != 0){
				myFileBits << i << ",";
			}
			if(toReportAll_param == 0 && tempVector[i] == 2){
				myFileBits << i << ",";
			}			
		}
		myFileBits << endl;
	}
	myFileBits.close();	
}

void updateQValues(Agent &agent, 
	vector<int> &Features, vector<float> &QValues, int option){
	for(int a = 0; a < agent.numberOfAvailActions; a++){
		float sumW = 0;
		for(unsigned int i = 0; i < Features.size(); i++){
			sumW += agent.w[option][a][Features[i]];
		}
		QValues[a] = sumW;
	}
}

int epsilonGreedy(Agent &agent, vector<float> &QValues, float epsilon){
	int randomActionTaken = 0;

	int action = Mathematics::argmax(QValues);
	//With probability epsilon: a <- random action in A(s)
	int random = rand();
	if(int(nearbyint(float(rand()%1000 < epsilon * 1000))) == 0){
		randomActionTaken = 1;
		action = rand() % agent.numberOfPrimitiveActions;
	}
	return action;
}

int playActionUpdatingAvg(ALEInterface& ale, Parameters *param, Agent &agent,
	RAMFeatures *ramFeatures, BPROFeatures *bproFeatures, int &frame, int nextAction, int iter){

	vector<bool> F(NUM_BITS, 0); //Set of active features
	vector<bool> Fprev;
	int reward = 0;

	//If the selected action was one of the primitive actions
	if(nextAction < agent.numberOfPrimitiveActions){ 
		for(int i = 0; i < param->numStepsPerAction; i++){
			reward += ale.act((Action) nextAction);
			frame++;
			Fprev.swap(F);
			F.clear();
			ramFeatures->getCompleteFeatureVector(ale.getRAM(), F);
			F.pop_back();
			updateAverage(param, agent, Fprev, F, frame, iter);
		}
	}
	//If the selected action was one of the options
	else{
		int currentAction;
		vector<int> Fbpro;	//Set of features active
		vector<float> Q(agent.numberOfPrimitiveActions, 0.0);    //Q(a) entries

		int option = nextAction - agent.numberOfPrimitiveActions;
		while(rand()%1000 > 1000 * param->optionTerminationProb && !ale.game_over()){
			//Get state and features active on that state:		
			Fbpro.clear();
			bproFeatures->getActiveFeaturesIndices(ale.getScreen(), Fbpro);
			updateQValues(agent, Fbpro, Q, option); //Update Q-values for each possible action
			currentAction = epsilonGreedy(agent, Q, param->epsilon);
			//Take action, observe reward and next state:
			reward += ale.act((Action) currentAction);
			frame++;
			Fprev.swap(F);
			F.clear();
			ramFeatures->getCompleteFeatureVector(ale.getRAM(), F);
			F.pop_back();
			updateAverage(param, agent, Fprev, F, frame, iter);
		}
	}
	return reward;
}

int playGame(ALEInterface& ale, Parameters *param, Agent &agent, 
	RAMFeatures *ramFeatures, BPROFeatures *bproFeatures, int iter){
	int score = 0;
	int frame = 0;
	int totalNumActions = agent.numberOfAvailActions;

	ale.reset_game();
	while(!ale.game_over()){
		int nextAction = rand() % totalNumActions;
		score += playActionUpdatingAvg(ale, param, agent, ramFeatures, bproFeatures, frame, nextAction, iter);
	}

	return score;
}

void gatherSamplesFromRandomTrajectories(ALEInterface& ale, Parameters *param, Agent &agent, int iter){
	cout << "Generating Samples to Identify Rare Events\n";
	RAMFeatures ramFeatures;
	BPROFeatures bproFeatures(param);
	for(int i = 1; i < param->numGamesToSampleRareEvents + 1; i++){
		int finalScore = playGame(ale, param, agent, &ramFeatures, &bproFeatures, iter);
		cout << i << ": Final score: " << finalScore << endl;
	}
}

void learnOptionsDerivedFromEigenEvents(){}