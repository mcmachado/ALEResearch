/****************************************************************************************
** Starting point for running the game to be controlled by a human player.
** If you want to save trajectories or the feature representation, you have to set the
** variables SAVE_TRAJECTORY and SAVE_REPRESENTATION properly, in the configuration file.
** This file uses RAM as the feature representation.
** 
** Author: Marlos C. Machado
***************************************************************************************/

#include <ale_interface.hpp>

#include "../../../../src/features/RAMFeatures.hpp"
#include "HumanPlayer.hpp"

int main(int argc, char** argv){
	//Using Basic features:
	RAMFeatures features;

	ALEInterface ale(1);
	ale.loadROM(argv[1]);
	//Instantiating the learning algorithm:
	HumanAgent human;
    //Learn a policy:
    human.evaluatePolicy(ale);
    return 0;
}