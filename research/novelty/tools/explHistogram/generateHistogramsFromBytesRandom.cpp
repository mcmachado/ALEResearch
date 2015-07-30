/****************************************************************************************
** Starting point for running the game to be controlled by a human player.
** 
** Author: Marlos C. Machado
***************************************************************************************/

#include <ale_interface.hpp>

int main(int argc, char** argv){
	int byteToRead;
	ALEInterface ale;

	if(argc != 5){
		printf("Usage: %s rom_file path_to_save_bits seed game_name\n", argv[0]);
		exit(1);
	}
	
	ale.loadROM(argv[1]);
	string outputFile = argv[2];
	int seed = atoi(argv[3]);
	string game_name = argv[4];
	
	srand(seed);
	ale.setBool("display_screen", false);
	ale.setBool("sound", false);
	ale.setInt("frame_skip", 5);
	ale.setInt("random_seed", seed);
	ale.setInt("max_num_frames_per_episode", 18000);
	ale.setFloat("repeat_action_prob", 0.00);	

	ActionVect actions = ale.getLegalActionSet();

	if(strcmp(game_name.c_str(), "freeway") == 0){
		byteToRead = 0x8E;
	} else if(strcmp(game_name.c_str(), "private_eye") == 0){
		byteToRead = 0xBE;
	} else{
		cout << "The game " << game_name << " is not supported.\n";
		exit(1);
	}


	int reward = 0;
    int count[0xFF];
    for(int i = 0; i < 0xFF; i++){
    	count[i] = 0;
    }

    ALERAM ram = ale.getRAM();
    unsigned int byte = ram.get(byteToRead);
    count[byte]++;

	while(!ale.game_over()) {
		reward += ale.act(actions[rand() % actions.size()]);
        ram = ale.getRAM();
        byte = ram.get(byteToRead);
        count[byte]++;
	}

	ofstream outFile;
    outFile.open(outputFile);

	for(int i = 0; i < 0xFF; i++){
		outFile << count[i] << endl;
	}
	outFile.close();

	printf("Episode ended with a score of %d points\n", reward);
    return 0;
}
