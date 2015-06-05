/****************************************************************************************
** Implementation of an agent that performs one single best action 1-epsilon percent of
** the time and epsilon percent of the time random actions. It was described in details 
** in the paper below.
**       "The Arcade Learning Environment: An Evaluation Platform for General Agents.
**        Marc G. Bellemare, Yavar Naddaf, Joel Veness, and Michael Bowling.
**        Journal of Artificial Intelligence Research, 47:253–279, 2013."
**
** Author: Marlos C. Machado
***************************************************************************************/

#include "PerturbAgent.hpp"
#include <climits>

PerturbAgent::PerturbAgent(Parameters *param){
	maxStepsInEpisode = param->getEpisodeLength();
	numEpisodesToEval = param->getNumEpisodesEval();
	epsilon = param->getEpsilon();
}

//I assumed one episode was representative.
//I don't want to run all actions for several episodes due to efficiency.
void PerturbAgent::learnPolicy(Environment<bool>& env){
	int reward = 0;
	int cumulativeReward = 0;
	int numActions;
	//It makes no sense to try ilegal actions:
	ActionVect actions;
	actions = env.getMinimalActionSet();
	numActions = actions.size();
	printf("Number of Actions: %d\n\n", numActions);
	int best = 0;
	int bestReward = INT_MIN;
	//For each action evaluate the return when executing only it:
	for(int a = 0; a < numActions; a++){
		int step = 0;
		while(!env.game_over() && step < maxStepsInEpisode) {
			reward = env.act(actions[a]);
			cumulativeReward += reward;
			step++;
		}
		//Keeping track of best action:
		if(cumulativeReward > bestReward){
			bestReward = cumulativeReward;
			best = a;
		}
		cumulativeReward = 0;
		env.reset_game();		
	}
	bestAction = best;
}

void PerturbAgent::evaluatePolicy(Environment<bool>& env){
	int reward = 0;
	int cumulativeReward = 0;
	//It makes no sense to try ilegal actions:
	ActionVect actions;
	actions = env.getMinimalActionSet();
	
	printf("Best Action: %d\n", bestAction);
	//Run numEpisodesToEval episodes with the agent always performing the same action
	//with 1 - epsilon probability, otherwise act randomly:
	for(int episode = 0; episode < numEpisodesToEval; episode++){
		int step = 0;
		while(!env.game_over() && step < maxStepsInEpisode) {
			if((rand() % 100) < 100 * (1 - epsilon)){
				//Act as the best action
				reward = env.act(actions[bestAction]);
			}
			else{
				//Act randomly
				reward = env.act(actions[rand()%actions.size()]);
			}
			cumulativeReward += reward;
			step++;
		}
		printf("Episode %d, Best Cumulative Reward: %d\n", episode + 1, cumulativeReward);
		cumulativeReward = 0;
		env.reset_game(); //Start the game again when the episode is over
	}
}

PerturbAgent::~PerturbAgent(){}
