/****************************************************************************************
** Implementation of an agent that acts randomly. It was described in details in the
** paper below.
**       "The Arcade Learning Environment: An Evaluation Platform for General Agents.
**        Marc G. Bellemare, Yavar Naddaf, Joel Veness, and Michael Bowling.
**        Journal of Artificial Intelligence Research, 47:253–279, 2013."
**
** Author: Marlos C. Machado
***************************************************************************************/

#include "RandomAgent.hpp"

RandomAgent::RandomAgent(Parameters *param){
	maxStepsInEpisode = param->getEpisodeLength();
	numEpisodesToEval = param->getNumEpisodesEval();
	useMinActions = param->isMinimalAction();
}

void RandomAgent::learnPolicy(Environment<bool>& env){
}

double RandomAgent::evaluatePolicy(Environment<bool>& env){
	int reward = 0;
	int totalReward = 0;
	int cumulativeReward = 0;
	int numActions;
	ActionVect actions;
	//Check if one wants to sample from all possible actions or only the valid ones:
	if(useMinActions){
		actions = env.getMinimalActionSet();
	}
	else{
		actions = env.getLegalActionSet();
	}
	numActions = actions.size();
	printf("Number of Actions: %d\n\n", numActions);
	//Repeat (for each episode):
	for(int episode = 0; episode < numEpisodesToEval; episode++){
		int step = 0;
		while(!env.game_over() && step < maxStepsInEpisode) {
			reward = env.act(actions[rand()%numActions]);
			cumulativeReward += reward;
			step++;
		}
		printf("Episode %d, Cumulative Reward: %d\n", episode + 1, cumulativeReward);
		totalReward += cumulativeReward;
		cumulativeReward = 0;
		env.reset_game(); //Start the game again when the episode is over
	}
	return double(totalReward)/numEpisodesToEval;
}

RandomAgent::~RandomAgent(){}
