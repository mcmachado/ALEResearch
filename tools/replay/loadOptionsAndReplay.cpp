#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <getopt.h>

#include "BPROFeatures.hpp"
#include "../../src/common/Graphics.hpp"

#define NUM_ARGS     9
#define NUM_ROWS    14
#define NUM_COLUMNS 16 
#define NUM_COLORS  128

#define STOCHASTICITY        0.00
#define PROB_TERMINATION     0.01
#define MAX_LENGTH_EPISODE   18000
#define NUM_STEPS_PER_ACTION 5

using namespace std;
string romPath;
string wgtPath;
int    seed;

ActionVect             actions;
vector<int>            F;		          //Set of features active
vector<float>          Q;                 //Q(a) entries
vector<float>          Qoptions;          //Q(a) entries
vector<string>         optionsPath;       //Paths for all files to be loaded
vector<vector<float> > w;                 //Theta, weights vector
vector<vector<vector<float> > > options;  //Theta, weights vector for options

//Algorithm related:
int currentAction;
float epsilon = 0.05;
//Environment related:
int numTotalActions, numFeatures;
int numBasicActions, numOptions;

void printHelp(char** argv){
	printf("Usage:    %s -s <SEED> -r <ROM> -w <WEIGHTS_LOAD> -n 3 <WEIGHT_1> <WEIGHT_2> <WEIGHT_3>\n", argv[0]);
	printf("   -s     %s[REQUIRED]%s seed to random number generator.\n", ANSI_COLOR_RED, ANSI_COLOR_RESET);
	printf("   -r     %s[REQUIRED]%s path to the rom to be played by the agent.\n", ANSI_COLOR_RED, ANSI_COLOR_RESET);
	printf("   -w     %s[REQUIRED]%s path to file containing the weights to be loaded.\n", ANSI_COLOR_RED, ANSI_COLOR_RESET);
	printf("   -n     %s[REQUIRED]%s number of options to be loaded.\n", ANSI_COLOR_RED, ANSI_COLOR_RESET);
	printf("   -h     print this help and exit\n");
	printf("\n");
}

void readParameters(int argc, char** argv){
	int option = 0;
	while ((option = getopt(argc, argv, "s:r:w:n:h")) != -1)
	{
		if (option == -1){
			break;
		}
		switch(option)
		{
			case 'h':
				printHelp(argv);
				exit(-1);
			case 'r':
				romPath = optarg;
				break;
			case 'w':
				wgtPath = optarg;
				break;
			case 's':
				seed = atoi(optarg);
				break;
			case 'n':
				numOptions = atoi(optarg);
				break;
			case ':':
         	case '?':
         		fprintf(stderr, "Try `%s -h' for more information.\n", argv[0]);
         		exit(-1);
			default:
				break;
				fprintf(stderr, "%s: invalid option %c\n", argv[0], option);
         		fprintf(stderr, "Try `%s -h' for more information.\n", argv[0]);
         		exit(-1);
		}
	}
	//Check if all parameters were properly set, otherwise interrupt	
	if(romPath.compare("") == 0 || wgtPath.compare("") == 0 
		|| seed == 0 || argc != NUM_ARGS + numOptions){
		printHelp(argv);
		exit(-1);
	}

	for(int i = 0; i < numOptions; i++){
		optionsPath.push_back(argv[NUM_ARGS + i]);
	}
}

void loadWeights(){
	int nActions, nFeatures;
	int i, j;
	float value;

	std::ifstream weightsFile (wgtPath.c_str());

	weightsFile >> nActions >> nFeatures;
	assert(nActions == numTotalActions);
	assert(nFeatures == numFeatures);

	while(weightsFile >> i >> j >> value){
		w[i][j] = value;
	}
}

void loadOptions(){
	for(int i = 0; i < numOptions; i++){
		string line;
		int nActions, nFeatures;
		int j, k;
		float value;

		std::ifstream weightsFile (optionsPath[i].c_str());

		weightsFile >> nActions >> nFeatures;
		assert(nActions == numBasicActions);
		assert(nFeatures == numFeatures);

		while(weightsFile >> j >> k >> value){
			options[i][j][k] = value;
		}
	}
}

void updateQValues(){
	for(int a = 0; a < numTotalActions; a++){
		float sumW = 0;
		for(unsigned int i = 0; i < F.size(); i++){
			sumW += w[a][F[i]];
		}
		Q[a] = sumW;
	}
}

void updateOptionQValues(int option){
	for(int a = 0; a < numBasicActions; a++){
		float sumW = 0;
		for(unsigned int i = 0; i < F.size(); i++){
			sumW += options[option][a][F[i]];
		}
		Qoptions[a] = sumW;
	}
}

int argmax(std::vector<float> array){
	assert(array.size() > 0);
	//Discover max value of the array:
	float max = array[0];
	for (unsigned int i = 0; i < array.size(); i++){
		if(max < array[i]){
			max = array[i];
		}
	}
	//We need to break ties, thus we save all  
	//indices that hold the same max value:
	std::vector<int> indices;
	for(unsigned int i = 0; i < array.size(); i++){
		if(fabs(array[i] - max) < 1e-10){
			indices.push_back(i);
		}
	}
	assert(indices.size() > 0);
	//Now we randomly pick one of the best
	return indices[rand()%indices.size()];
}

int epsilonGreedy(){
	int action = argmax(Q);
	//With probability epsilon: a <- random action in A(s)
	int random = rand();
	if((random % int(nearbyint(1.0/epsilon))) == 0) {
		action = rand() % numBasicActions;
	}
	return action;
}

int epsilonGreedyOption(){
	int action = argmax(Qoptions);
	//With probability epsilon: a <- random action in A(s)
	int random = rand();
	if((random % int(nearbyint(1.0/epsilon))) == 0) {
		action = rand() % numBasicActions;
	}
	return action;
}

int takeAction(ALEInterface& ale, BPROFeatures features, int actionToTake){
	int totalReward = 0;
	//If the selected action was one of the primitive actions
	if(actionToTake < numBasicActions){
		totalReward += ale.act(actions[actionToTake]);
	} 
	else{
		int currentAction;
		int option = actionToTake - numBasicActions;
		while(rand()%1000 > 1000 * PROB_TERMINATION && !ale.game_over()){
			F.clear();
			features.getActiveFeaturesIndices(ale.getScreen(), ale.getRAM(), F);
			updateOptionQValues(option);  //Update Q-values for each possible action of an option
			currentAction = epsilonGreedyOption();
			//Take action, observe reward and next state:
			totalReward += ale.act(actions[currentAction]);
		}
	}
	return totalReward;
}

int main(int argc, char** argv){

	readParameters(argc, argv);
	srand(seed);

	numBasicActions = 18;
	numTotalActions = numBasicActions + numOptions;
	numFeatures = NUM_COLUMNS * NUM_ROWS * NUM_COLORS 
					+ (2 * NUM_COLUMNS - 1) * (2 * NUM_ROWS - 1) * NUM_COLORS * NUM_COLORS + 1;

	for(int i = 0; i < numTotalActions; i++){
		//Initialize Q;
		Q.push_back(0);
		w.push_back(vector<float>(numFeatures, 0.0));
	}

	for(int i = 0; i < numOptions; i++){
		options.push_back(vector< vector<float> >(numBasicActions, vector<float>(numFeatures, 0.0)));
	}

	for(int i = 0; i < numBasicActions; i++){
		Qoptions.push_back(0);
	}

	loadWeights();
	loadOptions();
	int reward = 0;

	//Initializing ALE:
	ALEInterface ale;

	ale.setBool("display_screen", true);
	ale.setBool("sound", false);
	ale.setFloat("frame_skip", NUM_STEPS_PER_ACTION);
	ale.setFloat("stochasticity", STOCHASTICITY);
	ale.setInt("random_seed", seed);
	ale.setInt("max_num_frames_per_episode", MAX_LENGTH_EPISODE);

	/*
	std::string recordPath = "record";
    std::cout << std::endl;

    // Set record flags
    ale.setString("record_screen_dir", recordPath.c_str());
    ale.setString("record_sound_filename", (recordPath + "/sound.wav").c_str());
    // We set fragsize to 64 to ensure proper sound sync 
    ale.setInt("fragsize", 64);

    // Not completely portable, but will work in most cases
    std::string cmd = "mkdir ";
    cmd += recordPath; 
    system(cmd.c_str());
	*/

    ale.loadROM(romPath.c_str());
	//Initializing useful things to agent:
	BPROFeatures features;
	actions     = ale.getLegalActionSet();

	while(!ale.game_over()){
		//Get state and features active on that state:		
		F.clear();
		features.getActiveFeaturesIndices(ale.getScreen(), ale.getRAM(), F);
		updateQValues();       //Update Q-values for each possible action
		currentAction = epsilonGreedy();
		//Take action, observe reward and next state:
		reward += takeAction(ale, features, currentAction);
	}

	printf("Final score: %d\n", reward);

	return 0;
}