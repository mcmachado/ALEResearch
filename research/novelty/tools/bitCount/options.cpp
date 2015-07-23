#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <getopt.h>

#include "BPROFeatures.hpp"
#include "RAMFeatures.hpp"
#include "../../../../src/common/Graphics.hpp"

#define NUM_ARGS    13
#define NUM_ROWS    14
#define NUM_COLUMNS 16 
#define NUM_COLORS  128

#define STOCHASTICITY        0.00
#define PROB_TERMINATION     0.01
#define MAX_LENGTH_EPISODE   18000
#define NUM_STEPS_PER_ACTION 5
#define EPSILON              0.05

using namespace std;

//Parameters:
int    seed;
ofstream outFile;
string gameName;
string romPath;
string wgtPath;
string outputFile;
vector<string> optionsPath; //Paths for all files to be loaded

ActionVect actions;

vector<bool> Fram;
RAMFeatures ramFeatures;

//Algorithm related:
int currentAction;
//Environment related:
int numTotalActions, numFeatures;
int numBasicActions, numOptions;

void printHelp(char** argv){
	printf("Usage:    %s -s <SEED> -r <ROM> -w <WEIGHTS_LOAD> -g <GAME_NAME> -n 3 <WEIGHT_1> <WEIGHT_2> <WEIGHT_3>\n", argv[0]);
	printf("   -s     %s[REQUIRED]%s seed to random number generator.\n", ANSI_COLOR_RED, ANSI_COLOR_RESET);
	printf("   -r     %s[REQUIRED]%s path to the rom to be played by the agent.\n", ANSI_COLOR_RED, ANSI_COLOR_RESET);
	printf("   -w     %s[REQUIRED]%s path to file containing the weights to be loaded.\n", ANSI_COLOR_RED, ANSI_COLOR_RESET);
	printf("   -g     %s[REQUIRED]%s game name.\n", ANSI_COLOR_RED, ANSI_COLOR_RESET);
	printf("   -o     %s[REQUIRED]%s output file to save RAM states.\n", ANSI_COLOR_RED, ANSI_COLOR_RESET);
	printf("   -n     %s[REQUIRED]%s number of options to be loaded.\n", ANSI_COLOR_RED, ANSI_COLOR_RESET);
	printf("   -h     print this help and exit\n");
	printf("\n");
}

void readParameters(int argc, char** argv){
	int option = 0;
	while ((option = getopt(argc, argv, "s:r:w:n:g:o:h")) != -1)
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
			case 'g':
				gameName = optarg;
				break;
			case 'o':
				outputFile = optarg;
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

	if(numOptions > 0){
		for(int i = 0; i < numOptions; i++){
			int idx = argc - i - 1;
			optionsPath.push_back(argv[idx]);
		}
	}
}

void loadWeights(vector<vector<float> > &w){
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

void loadOptions(vector<vector<vector<float> > > &learnedOptions){
	
	for(int i = 0; i < numOptions; i++){
		learnedOptions.push_back(vector< vector<float> >(numBasicActions, vector<float>(numFeatures, 0.0)));
	}

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
			learnedOptions[i][j][k] = value;
		}
	}
}

void updateQValues(vector<int> &Features, vector<float> &QValues, vector<vector<float> > &w){
	for(int a = 0; a < QValues.size(); a++){
		float sumW = 0;
		for(unsigned int i = 0; i < Features.size(); i++){
			sumW += w[a][Features[i]];
		}
		QValues[a] = sumW;
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

int epsilonGreedy(vector<float> &QValues){
	int action = argmax(QValues);
	//With probability epsilon: a <- random action in A(s)
	int random = rand();
	if((random % int(nearbyint(1.0/EPSILON))) == 0) {
		action = rand() % numTotalActions;
	}
	return action;
}

int playOption(ALEInterface& ale, BPROFeatures features, int option, 
	vector<vector<vector<float> > > &learnedOptions){

	vector<int> F;	//Set of features active
	vector<float> QOptions(numBasicActions, 0.0);    //Q(a) entries
	int cumReward = 0;

	while(rand()%1000 > 1000 * PROB_TERMINATION && !ale.game_over()){
		F.clear();
		features.getActiveFeaturesIndices(ale.getScreen(), ale.getRAM(), F);

		updateQValues(F, QOptions, learnedOptions[option]);
		currentAction = argmax(QOptions);

		//With probability epsilon: a <- random action in A(s)
		int random = rand();
		if((random % int(nearbyint(1.0/EPSILON))) == 0) {
			currentAction = rand() % numBasicActions;
		}
		//Take action, observe reward and next state:
		cumReward += ale.act(actions[currentAction]);

		Fram.clear();
        ramFeatures.getCompleteFeatureVector(ale.getRAM(), Fram);

        for(int i = 0; i < Fram.size(); i++){
            outFile << Fram[i] << ",";
        }
        outFile << endl;		
	}
	return cumReward;
}

int takeAction(ALEInterface& ale, BPROFeatures features, int actionToTake,
	vector<vector<vector<float> > > &learnedOptions){
	int totalReward = 0;
	//If the selected action was one of the primitive actions
	if(actionToTake < numBasicActions){
		totalReward += ale.act(actions[actionToTake]);
		Fram.clear();
        ramFeatures.getCompleteFeatureVector(ale.getRAM(), Fram);

        for(int i = 0; i < Fram.size(); i++){
            outFile << Fram[i] << ",";
        }
        outFile << endl;		
	} 
	else{
		int option = actionToTake - numBasicActions;
		totalReward = playOption(ale, features, option, learnedOptions);
	}
	return totalReward;
}

int main(int argc, char** argv){
	
	readParameters(argc, argv);
	srand(seed);

    outFile.open(outputFile);

	int reward = 0;
	//Initializing useful things to agent:
	BPROFeatures features(gameName);

	numBasicActions = 18;
	numTotalActions = numBasicActions + numOptions;
	numFeatures = NUM_COLUMNS * NUM_ROWS * NUM_COLORS 
					+ (2 * NUM_COLUMNS - 1) * (2 * NUM_ROWS - 1) * NUM_COLORS * NUM_COLORS + 1;

	vector<int> F;	//Set of features active
	vector<float> Q(numTotalActions, 0); //Q(a) entries
	vector<vector<float> > w(numTotalActions, vector<float>(numFeatures, 0.0)); //Theta, weights vector
	vector<vector<vector<float> > > learnedOptions; //Theta, weights vector for options

	loadWeights(w);
	loadOptions(learnedOptions);

	//Initializing ALE:
	ALEInterface ale;
	ale.setBool("display_screen", true);
	ale.setBool("sound", false);
	ale.setFloat("frame_skip", NUM_STEPS_PER_ACTION);
	ale.setFloat("repeat_action_prob", STOCHASTICITY);
	ale.setInt("random_seed", seed);
	ale.setInt("max_num_frames_per_episode", MAX_LENGTH_EPISODE);

    ale.loadROM(romPath.c_str());

    Fram.clear();
    ramFeatures.getCompleteFeatureVector(ale.getRAM(), Fram);

    for(int i = 0; i < Fram.size(); i++){
        outFile << Fram[i] << ",";
    }
    outFile << endl;

	actions = ale.getLegalActionSet();
	while(!ale.game_over()){
		//Get state and features active on that state:		
		F.clear();
		features.getActiveFeaturesIndices(ale.getScreen(), ale.getRAM(), F);
		updateQValues(F, Q, w);       //Update Q-values for each possible action
		currentAction = epsilonGreedy(Q);
		//Take action, observe reward and next state:
		reward += takeAction(ale, features, currentAction, learnedOptions);
	}

	printf("Episode ended with a score of %d points\n", reward);
	outFile.close();
	return 0;
}
