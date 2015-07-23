/****************************************************************************************
** Starting point for running the game to be controlled by a human player.
** 
** Author: Marlos C. Machado
***************************************************************************************/

#include <ale_interface.hpp>
#include "RAMFeatures.hpp"

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
	srand(seed);

	ActionVect actions = ale.getLegalActionSet();
    RAMFeatures features;
    vector<bool> F;

    ofstream outFile;
    outFile.open(outputFile);

	int reward = 0;
    int step = 0;
	
	F.clear();
    features.getCompleteFeatureVector(ale.getRAM(), F);
    for(int i = 0; i < F.size(); i++){
        outFile << F[i] << ",";
    }
    outFile << endl;

	while(!ale.game_over()) {
		reward += ale.act(actions[rand() % actions.size()]);
        F.clear();
        features.getCompleteFeatureVector(ale.getRAM(), F);

        for(int i = 0; i < F.size(); i++){
            outFile << F[i] << ",";
        }
        outFile << endl;
		step++;
	}
	printf("Episode ended with a score of %d points\n", reward);
    return 0;
}
