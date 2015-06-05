/****************************************************************************************
** Implementation of an agent that acts randomly. It was described in details in the
** paper below.
**       "The Arcade Learning Environment: An Evaluation Platform for General Agents.
**        Marc G. Bellemare, Yavar Naddaf, Joel Veness, and Michael Bowling.
**        Journal of Artificial Intelligence Research, 47:253–279, 2013."
**
** Author: Marlos C. Machado
***************************************************************************************/

#ifndef RANDAGENT_H
#define RANDAGENT_H
#include "../Agent.hpp"

class RandomAgent : public Agent<bool>{
	private:
		int maxStepsInEpisode;
		int numEpisodesToEval;
		int useMinActions;
		/** 
		* Constructor to prohibit the creation of an object without the parameters values.
		*/
		RandomAgent();
	public:
		/** 
		* Constructor to store parameters information for object's creation
		*
		* @param Parameters *param parameters read from the configuration file
		*/
		RandomAgent(Parameters *param);
		/**
 		* This method is useless for this agent. To act randomly no learning is required.
 		*
 		* @param Environment<bool>& env Arcade Learning Environment interface: object used to define agents'
 		*        actions, obtain simulator screen, RAM, etc.
 		*/
        void learnPolicy(Environment<bool>& env);
		/**
 		* Implementation of an agent that selects actions randomly.
 		*
 		* @param Environment<bool>& env Arcade Learning Environment interface: object used to define agents'
 		*        actions, obtain simulator screen, RAM, etc.
 		*/
		void evaluatePolicy(Environment<bool>& env);
		/**
		* Destructor, not necessary in this class.
		*/
		~RandomAgent();
};


#endif
