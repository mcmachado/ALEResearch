/****************************************************************************************
** Implementation of an agent that performs one single action, described in details in 
** the paper below.
**       "The Arcade Learning Environment: An Evaluation Platform for General Agents.
**        Marc G. Bellemare, Yavar Naddaf, Joel Veness, and Michael Bowling.
**        Journal of Artificial Intelligence Research, 47:253–279, 2013."
**
** Author: Marlos C. Machado
***************************************************************************************/

#ifndef AGENT_H
#define AGENT_H
#include "../Agent.hpp"
#endif

class ConstantAgent : public Agent{
	private:
		int bestAction;
		int maxStepsInEpisode;
		int numEpisodesToEval;
		/** 
		* Constructor to prohibit the creation of an object without the parameters values.
		*/
		ConstantAgent();
	public:
		/** 
		* Constructor to store parameters information for object's creation
		*
		* @param Parameters *param parameters read from the configuration file
		*/
		ConstantAgent(Parameters *param);
		/**
 		* The agent learns the best action to be executed in the future.
 		* This method returns the action that obtains a larger return after a whole episode.
 		* It is implemented assuming that the "best" action is consistent for multiple runs,
 		* therefore each action is tried for one complete episode and the action with maximum
 		* return is returned.
 		*
 		* @param ALEInterface& ale Arcade Learning Environment interface: object used to define agents'
 		*        actions, obtain simulator screen, RAM, etc.
 		* @param Features *features object that defines what feature function that will be used by the RL
 		*        agents. Since no features are relevant here this parameter is set to null by default.
 		*/
		void learnPolicy(ALEInterface& ale, Features *features = NULL);
		/**
 		* Implementation of an agent that selects the action that obtains a higher return every time.
 		* It is run for a given number of episodes.
 		*
 		* @param ALEInterface& ale Arcade Learning Environment interface: object used to define agents'
 		*        actions, obtain simulator screen, RAM, etc.
 		* @param Features *features object that defines what feature function that will be used by the RL
 		*        agents. Since no features are relevant here this parameter is set to null by default.
 		*/
		void evaluatePolicy(ALEInterface& ale, Features *features = NULL);
		/**
		* Destructor, not necessary in this class.
		*/
		~ConstantAgent();
};
