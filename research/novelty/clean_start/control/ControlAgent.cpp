/* Author: Marlos C. Machado */

#include "ControlAgent.hpp"
#include "../observations/RAMFeatures.hpp"
#include "../observations/BPROFeatures.hpp"

/*
int playActionUpdatingAvg(ALEInterface& ale, RAMFeatures *ram, BPROFeatures *features, int nextAction, int gameId){
	vector<bool> F(NUM_BITS, 0); //Set of active features
	vector<bool> Fprev;
	int reward = 0;

	//If the selected action was one of the primitive actions
	if(nextAction < NUM_ACTIONS){ 
		for(int i = 0; i < FRAME_SKIP; i++){
			reward += ale.act((Action) nextAction);
			frame++;
			Fprev.swap(F);
			F.clear();
			ram->getCompleteFeatureVector(ale.getScreen(), ale.getRAM(), F);
			F.pop_back();
			updateAverage(Fprev, F, frame, gameId);
		}
	}
	//If the selected action was one of the options
	else{
		int currentAction;
		vector<int> Fbpro;	                  //Set of features active
		vector<float> Q(NUM_ACTIONS, 0.0);    //Q(a) entries

		int option = nextAction - NUM_ACTIONS;
		while(rand()%1000 > 1000 * PROB_TERMINATION && !ale.game_over()){
			//Get state and features active on that state:		
			Fbpro.clear();
			features->getActiveFeaturesIndices(ale.getScreen(), ale.getRAM(), Fbpro);
			updateQValues(Fbpro, Q, option);       //Update Q-values for each possible action
			currentAction = epsilonGreedy(Q);
			//Take action, observe reward and next state:
			reward += ale.act((Action) currentAction);
			frame++;
			Fprev.swap(F);
			F.clear();
			ram->getCompleteFeatureVector(ale.getScreen(), ale.getRAM(), F);
			F.pop_back();
			updateAverage(Fprev, F, frame, gameId);
		}
	}
	return reward;
}
*/
int playGame(ALEInterface& ale, RAMFeatures *ram, BPROFeatures *bpro){
	int score = 0;
	//int totalNumActions = agent.getNumberOfAvailActions();
	int totalNumActions = 10; //I'll change it later, to the line above

	ale.reset_game();
	while(!ale.game_over()){
		int nextAction = rand() % totalNumActions;
		//score += playActionUpdatingAvg(ale, ram, bpro, nextAction);
	}

	return score;
}

void gatherSamplesFromRandomTrajectories(ALEInterface& ale, Parameters *param){
	cout << "Generating Samples to Identify Rare Events\n";
	RAMFeatures ramFeatures;
	BPROFeatures bproFeatures(param);
	for(int i = 1; i < param->numGamesToSampleRareEvents + 1; i++){
		int finalScore = playGame(ale, &ramFeatures, &bproFeatures);
		cout << i << ": Final score: " << finalScore << endl;
	}
}

void learnOptionsDerivedFromEigenEvents(){}