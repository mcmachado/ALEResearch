/****************************************************************************************
** Starting point for running the game to be controlled by a human player.
** 
** Author: Marlos C. Machado
***************************************************************************************/

#include <ale_interface.hpp>

#include "HumanPlayer.hpp"

int main(int argc, char** argv){
	ALEInterface ale(1);

	if(argc != 4){
		printf("Usage: %s rom_file path_to_save_bits seed\n", argv[0]);
		exit(1);
	}

	int seed = atoi(argv[3]);

	ale.setInt("frame_skip", 5);
	ale.setInt("random_seed", seed);
	ale.setInt("max_num_frames_per_episode", 18000);
	ale.setFloat("repeat_action_prob", 0.00);

	ale.loadROM(argv[1]);
	string outputFile = argv[2];

	HumanAgent human;
    human.evaluatePolicy(ale, outputFile);
    return 0;
}
