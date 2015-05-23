/****************************************************************************************
** Implementation of an agent that performs one single action, described in details in 
** the paper below.
**       "The Arcade Learning Environment: An Evaluation Platform for General Agents.
**        Marc G. Bellemare, Yavar Naddaf, Joel Veness, and Michael Bowling.
**        Journal of Artificial Intelligence Research, 47:253–279, 2013."
**
** Author: Marlos C. Machado
***************************************************************************************/

#include "ConstantAgent.hpp"
#include <climits>

ConstantAgent::ConstantAgent(Parameters *param){
	maxStepsInEpisode = param->getEpisodeLength();
	numEpisodesToEval = param->getNumEpisodesEval();
}

//I assumed one episode was representative.
//I don't want to run all actions for several episodes due to efficiency.
void ConstantAgent::learnPolicy(ALEInterface& ale, Features *features){
	int reward = 0;
	int cumulativeReward = 0;
	int numActions;
	//It makes no sense to try ilegal actions:
	ActionVect actions;
	actions = ale.getMinimalActionSet();
	numActions = actions.size();
	printf("Number of Actions: %d\n\n", numActions);
	int best = 0;
	int bestReward = INT_MIN;
	//For each action evaluate the return when executing only it:
	for(int a = 0; a < numActions; a++){
		int step = 0;
		while(!ale.game_over() && step < maxStepsInEpisode) {
			reward = ale.act(actions[a]);
			cumulativeReward += reward;
			step++;
		}
		//Keeping track of best action:
		if(cumulativeReward > bestReward){
			bestReward = cumulativeReward;
			best = a;
		}
		printf("Action %d, Cumulative Reward: %d\n", a, cumulativeReward);
		cumulativeReward = 0;
		ale.reset_game(); //Start the game again when the episode is over
	}
	bestAction = best;
}

void ConstantAgent::evaluatePolicy(ALEInterface& ale, Features *features){
	int reward = 0;
	int cumulativeReward = 0;
	//It makes no sense to try ilegal actions:
	ActionVect actions;
	actions = ale.getMinimalActionSet();
	
	printf("Best Action: %d\n", bestAction);
	//Run numEpisodesToEval episodes with the agent always performing the same action:
	for(int episode = 0; episode < numEpisodesToEval; episode++){
		int step = 0;
		while(!ale.game_over() && step < maxStepsInEpisode) {
			reward = ale.act(actions[bestAction]);
			cumulativeReward += reward;
			step++;
		}
		printf("Episode %d, Best Cumulative Reward: %d\n", episode + 1, cumulativeReward);
		cumulativeReward = 0;
		ale.reset_game(); //Start the game again when the episode is over
	}
}

ConstantAgent::~ConstantAgent(){}
