/* Author: Marlos C. Machado */

#include "ControlAgent.hpp"
#include "../common/Mathematics.hpp"
#include "../observations/RAMFeatures.hpp"
#include "../observations/BPROFeatures.hpp"

#define NUM_BITS 1024

void updateAverage(vector<bool> Fprev, vector<bool> F, int frame){}

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
	RAMFeatures *ramFeatures, BPROFeatures *bproFeatures, int &frame, int nextAction){

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
			updateAverage(Fprev, F, frame);
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
			updateAverage(Fprev, F, frame);
		}
	}
	return reward;
}

int playGame(ALEInterface& ale, Parameters *param, Agent &agent, RAMFeatures *ramFeatures, BPROFeatures *bproFeatures){
	int score = 0;
	int frame = 0;
	int totalNumActions = agent.numberOfAvailActions;

	ale.reset_game();
	while(!ale.game_over()){
		int nextAction = rand() % totalNumActions;
		score += playActionUpdatingAvg(ale, param, agent, ramFeatures, bproFeatures, frame, nextAction);
	}

	return score;
}

void gatherSamplesFromRandomTrajectories(ALEInterface& ale, Parameters *param, Agent &agent){
	cout << "Generating Samples to Identify Rare Events\n";
	RAMFeatures ramFeatures;
	BPROFeatures bproFeatures(param);
	for(int i = 1; i < param->numGamesToSampleRareEvents + 1; i++){
		int finalScore = playGame(ale, param, agent, &ramFeatures, &bproFeatures);
		cout << i << ": Final score: " << finalScore << endl;
	}
}

void learnOptionsDerivedFromEigenEvents(){}