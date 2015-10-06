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

#ifndef PERTURBAGENT_H
#define PERTURBAGENT_H

#include "../Agent.hpp"

class PerturbAgent : public Agent<bool>{
	private:
		int bestAction;
		int maxStepsInEpisode;
		int numEpisodesToEval;
		double epsilon;
		/** 
		* Constructor to prohibit the creation of an object without the parameters values.
		*/
		PerturbAgent();
	public:
		/** 
		* Constructor to store parameters information for object's creation
		*
		* @param Parameters *param parameters read from the configuration file
		*/
		PerturbAgent(Parameters *param);
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
 		* Implementation of an agent that selects the action that obtains a higher return 1 - epsilon 
 		* percent of the time and epsilon percent of the time acts randomly.
 		*
		* @param Environment<bool>& env Arcade Learning Environment interface: object used to define agents'
 		*        actions, obtain simulator screen, RAM, etc.
 		*/
		double evaluatePolicy(Environment<bool>& env);
		/**
		* Destructor, not necessary in this class.
		*/
		~PerturbAgent();
};

#endif
