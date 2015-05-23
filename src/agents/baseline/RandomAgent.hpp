/****************************************************************************************
** Implementation of an agent that acts randomly. It was described in details in the
** paper below.
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

class RandomAgent : public Agent{
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
 		* @param ALEInterface& ale Arcade Learning Environment interface: object used to define agents'
 		*        actions, obtain simulator screen, RAM, etc.
 		* @param Features *features object that defines what feature function that will be used by the RL
 		*        agents. Since no features are relevant here this parameter is set to null by default.
 		*/
		void learnPolicy(ALEInterface& ale, Features *features = NULL);
		/**
 		* Implementation of an agent that selects actions randomly.
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
		~RandomAgent();
};
