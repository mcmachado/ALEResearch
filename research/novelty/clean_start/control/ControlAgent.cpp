/* Author: Marlos C. Machado */

#include "ControlAgent.hpp"

#define NUM_BITS 1024

int playGame(ALEInterface& ale, Parameters *param, Agent &agent, int iter){
	int score = 0;
	int frame = 0;
	int totalNumActions = agent.numberOfAvailActions;

	ale.reset_game();
	while(!ale.game_over()){
		int nextAction = rand() % totalNumActions;
		score += agent.playActionUpdatingAvg(ale, param, frame, nextAction, iter);
	}

	return score;
}

void gatherSamplesFromRandomTrajectories(ALEInterface& ale, Parameters *param, Agent &agent, int iter){
	cout << "Generating Samples to Identify Rare Events\n";
	for(int i = 1; i < param->numGamesToSampleRareEvents + 1; i++){
		int finalScore = playGame(ale, param, agent, iter);
		cout << i << ": Final score: " << finalScore << endl;
	}
}

void learnOptionsDerivedFromEigenEvents(){}