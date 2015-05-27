#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <getopt.h>

#include "BPROFeatures.hpp"
#include "../../src/common/Graphics.hpp"

#define NUM_ROWS    14
#define NUM_COLUMNS 16 
#define NUM_COLORS  128

#define STOCHASTICITY        0.00
#define MAX_LENGTH_EPISODE   18000
#define NUM_STEPS_PER_ACTION 5

using namespace std;
string romPath;
string wgtPath;
int    seed;

ActionVect              actions;
vector<int>             F;		     //Set of features active
vector<double>          Q;           //Q(a) entries
vector<vector<double> > w;           //Theta, weights vector

//Algorithm related:
int currentAction;
float epsilon = 0.05;
//Environment related:
int numActions, numFeatures;

void printHelp(char** argv){
	printf("Usage:    %s[OPTIONS]\n", argv[0]);
	printf("   -s     %s[REQUIRED]%s seed to random number generator.\n", ANSI_COLOR_RED, ANSI_COLOR_RESET);
	printf("   -r     %s[REQUIRED]%s path to the rom to be played by the agent.\n", ANSI_COLOR_RED, ANSI_COLOR_RESET);
	printf("   -w     %s[REQUIRED]%s path to file containing the weights to be loaded.\n", ANSI_COLOR_RED, ANSI_COLOR_RESET);
	printf("   -h     print this help and exit\n");
	printf("\n");
}

void readParameters(int argc, char** argv){
	int option = 0;
	while ((option = getopt(argc, argv, "s:r:w:h")) != -1)
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
	if(romPath.compare("") == 0 || wgtPath.compare("") == 0 || seed == 0){
		printHelp(argv);
		exit(-1);
	}
}

void loadWeights(string pathWeightsFileToLoad){
	int nActions, nFeatures;
	int i, j;
	double value;

	std::ifstream weightsFile (pathWeightsFileToLoad.c_str());

	weightsFile >> nActions >> nFeatures;
	assert(nActions == numActions);
	assert(nFeatures == numFeatures);

	while(weightsFile >> i >> j >> value){
		w[i][j] = value;
	}
}

void updateQValues(){
	for(int a = 0; a < numActions; a++){
		double sumW = 0;
		for(unsigned int i = 0; i < F.size(); i++){
			sumW += w[a][F[i]];
		}
		Q[a] = sumW;
	}
}

int argmax(std::vector<double> array){
	assert(array.size() > 0);
	//Discover max value of the array:
	double max = array[0];
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
		action = rand() % numActions;
	}
	return action;
}

int main(int argc, char** argv){

	readParameters(argc, argv);
	srand(seed);

	//Initializing ALE:
	ALEInterface ale(1);
	ale.setFloat("frame_skip", NUM_STEPS_PER_ACTION);
	ale.setFloat("stochasticity", STOCHASTICITY);
	ale.setInt("random_seed", seed);
	ale.setInt("max_num_frames_per_episode", MAX_LENGTH_EPISODE);
	ale.loadROM(romPath.c_str());

	//Initializing useful things to agent:
	BPROFeatures features;
	actions     = ale.getLegalActionSet();
	numActions  = actions.size();
	numFeatures = NUM_COLUMNS * NUM_ROWS * NUM_COLORS 
					+ (2 * NUM_COLUMNS - 1) * (2 * NUM_ROWS - 1) * NUM_COLORS * NUM_COLORS + 1;

	for(int i = 0; i < numActions; i++){
		//Initialize Q;
		Q.push_back(0);
		w.push_back(vector<double>(numFeatures, 0.0));
	}

	loadWeights(wgtPath);
	int reward = 0;

	while(!ale.game_over()){
		//Get state and features active on that state:		
		F.clear();
		features.getActiveFeaturesIndices(ale.getScreen(), ale.getRAM(), F);
		updateQValues();       //Update Q-values for each possible action
		currentAction = epsilonGreedy();
		//Take action, observe reward and next state:
		reward += ale.act(actions[currentAction]);
	}

	printf("Final score: %d\n", reward);

	return 0;
}