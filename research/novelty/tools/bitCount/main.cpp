/****************************************************************************************
** Starting point for running the game to be controlled by a human player.
** 
** Author: Marlos C. Machado
***************************************************************************************/

#include <ale_interface.hpp>

#include "HumanPlayer.hpp"

int main(int argc, char** argv){
	ALEInterface ale(1);

	if(argc != 3){
		printf("Usage: %s rom_file path_to_save_bits\n", argv[0]);
		exit(1);
	}

	ale.loadROM(argv[1]);
	string outputFile = argv[2];

	HumanAgent human;
    human.evaluatePolicy(ale, outputFile);
    return 0;
}
