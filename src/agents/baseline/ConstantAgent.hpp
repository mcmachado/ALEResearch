/****************************************************************************************
** Implementation of an agent that performs one single action, described in details in 
** the paper below.
**       "The Arcade Learning Environment: An Evaluation Platform for General Agents.
**        Marc G. Bellemare, Yavar Naddaf, Joel Veness, and Michael Bowling.
**        Journal of Artificial Intelligence Research, 47:253–279, 2013."
**
** Author: Marlos C. Machado
***************************************************************************************/

#ifndef CONSTAGENT_H
#define CONSTAGENT_H
#include "../Agent.hpp"

class ConstantAgent : public Agent<bool>{
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
 		* @param Environment<bool>& env Arcade Learning Environment interface: object used to define agents'
 		*        actions, obtain simulator screen, RAM, etc.
 		*/
		void learnPolicy(Environment<bool>& env);
		/**
 		* Implementation of an agent that selects the action that obtains a higher return every time.
 		* It is run for a given number of episodes.
 		*
 		* @param Environment<bool>& env Arcade Learning Environment interface: object used to define agents'
 		*        actions, obtain simulator screen, RAM, etc.
 		*/
		double evaluatePolicy(Environment<bool>& env);
		/**
		* Destructor, not necessary in this class.
		*/
		~ConstantAgent();
};



#endif
