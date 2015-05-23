/****************************************************************************************
** Implementation of an agent that is controlled by a human player.
**
** TODO: More tests need to be done regarding FPS. Right now there is no concern with it.
**       Additionally, saving trajectories is very slow (saving features not that much),
**       is there anyway we can optimize it??
** 
** Author: Marlos C. Machado
***************************************************************************************/

#ifndef AGENT_H
#define AGENT_H
#include "../Agent.hpp"
#endif

class HumanAgent : public Agent{
	private:
		int maxStepsInEpisode;
		int numEpisodesToEval;
		bool toSaveRepr;
		bool toSaveTrajectory;
		std::string featReprFile;
    	std::string trajectoryFile;

		/**
		* This method receives the action from the keyboard and returns it.
		* 
		* REMARKS: This method was extracted from SDLKeyboardAgent in the ALE.
		*
		* @return int action to be performed.
		*/
	#ifdef __USE_SDL			
		Action receiveAction();
		/**
		* This method saves in a file specified in featReprFile the feature representation of 
		* the agent's current state along the game.
		*
		* TODO: This method may be very slow according to the feature representation selected.
		*       Is there any optimization that can be done?
		*
 		* @param ALEInterface& ale Arcade Learning Environment interface: object used to define agents'
 		*        actions, obtain simulator screen, RAM, etc. 
 		* @param Features *features object that defines what feature function that will be used by the RL
 		*        agents. Since no features are relevant here this parameter is set to null by default.		
		*/
		void saveFeatureRepr(ALEInterface& ale, Features *features);
		/**
		* This method saves in a file specified in trajectoryFile the agent's
		* trajectory (state, action, reward). It is only called if the variable
		* toSaveTrajectory is defined as TRUE.
		*
		* TODO: This method is extremely slow. Is there any optimization that makes it feasible?
		*
 		* @param int reward the reward obtained with the action taken in the current state
		*/
		void saveTrajectory(int reward);
		/**
		* This method saves in a file specified in trajectoryFile the agent's
		* trajectory (state, action, reward). It is only called if the variable
		* toSaveTrajectory is defined as TRUE.
		*
		* TODO: This method is extremely slow. Is there any optimization that makes it feasible?
		*
 		* @param ALEInterface& ale Arcade Learning Environment interface: object used to define agents'
 		*        actions, obtain simulator screen, RAM, etc. 
 		* @param int takenAction the code for the actiont taken		
		*/
		void saveTrajectory(ALEInterface& ale, int takenAction);
#endif		
	public:
		/**
		* Constructor
		*
		* @param Parameters *param parameters read from the configuration file
		*/
		HumanAgent(Parameters *param);
		/**
 		* This method should be implemented due to the superclass Agent, however it is useless for human
 		* controlled agents.
 		*
 		* @param ALEInterface& ale Arcade Learning Environment interface: object used to define agents'
 		*        actions, obtain simulator screen, RAM, etc. 
 		* @param Features *features object that defines what feature function that will be used by the RL
 		*        agents. Since no features are relevant here this parameter is set to null by default.
 		*/
		void learnPolicy(ALEInterface& ale, Features *features = NULL);
		/**
 		* Implementation of an agent that is controlled by a human player.
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
		~HumanAgent();

};
