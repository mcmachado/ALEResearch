/****************************************************************************************
** Implementation of an agent that is controlled by a human player.
** 
** Author: Marlos C. Machado
***************************************************************************************/

#include <ale_interface.hpp>

class HumanAgent{
	private:
		int maxStepsInEpisode;

		/**
		* This method receives the action from the keyboard and returns it.
		* 
		* REMARKS: This method was extracted from SDLKeyboardAgent in the ALE.
		*
		* @return int action to be performed.
		*/
		Action receiveAction();
	public:
		/**
		* Constructor
		*/
		HumanAgent();

		/**
 		* Implementation of an agent that is controlled by a human player.
 		*
 		* @param ALEInterface& ale Arcade Learning Environment interface: object used to define agents'
 		*        actions, obtain simulator screen, RAM, etc.
 		*/
		void evaluatePolicy(ALEInterface& ale, string outputFile);
		/**
		* Destructor, not necessary in this class.
		*/
		~HumanAgent();

};
