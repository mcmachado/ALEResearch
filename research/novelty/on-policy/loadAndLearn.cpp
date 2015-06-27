/*TODO:
 1. Implement the Sarsa/Option learning policy
 2. Make the background to be transparent
 2. Test my own code...
*/

/******************************************************************************************
** In this code I load all the weights that were learned in the first round (different   **
** eigen-events) and then I am going to switch between them as primitive actions. Then I **
** generate the rare events again. This will be used for a second Sarsa implementation.  **
**                                                                                       **
** Author: Marlos C. Machado                                                             **
*******************************************************************************************/

#include <getopt.h>
#include <vector>

#include <ale_interface.hpp>

#include "features/BPROFeatures.hpp"
#include "../../../src/common/Graphics.hpp"
#include "../../../src/features/RAMFeatures.hpp"
#include "../../../src/common/Mathematics.hpp"

#define NUM_MIN_ARGS       13
//Features:
#define NUM_BITS         1024
//Agent's:
#define FRAME_SKIP          5
#define NUM_ACTIONS        18
#define PROB_TERMINATION 0.01
//Average:
#define FRAMES_TO_WAIT    300

int    frame = 0;

vector<float> frequency;            //[0:1023] transitions 0->1; [1024:2048] transitions 1->0
vector<string> optionsWgts;         //Vector containing the files with weights representing options

vector<vector<vector<float> > > w;  //Theta, weights vector

//Parameters:
string game_param;
string romPath_param;
string outputPath_param;

int seed_param  = 0;
int numOptions_param = -1;
int toReportAll_param = -1;
float freqThreshold_param = 0.0;

/**
* Prints the instructions related to the program's parameters in a friendly mode
*
* @param char** argv received from the command line
*/
void printHelp(char** argv){
	printf("Usage:    %s -r <ROM> -o <OUT_FILE> -t <THRESHOLD> -c <REPORT_IRR_TRANS> -n <NUM_OPTIONS> <OPTION_1> <OPTION_2> ... <OPTION_N> \n", argv[0]);
	printf("   -r     %s[REQUIRED]%s path to the rom to be played by the agent.\n", ANSI_COLOR_RED, ANSI_COLOR_RESET);
	printf("   -o     %s[REQUIRED]%s path to the file to be written with the output.\n", ANSI_COLOR_RED, ANSI_COLOR_RESET);
	printf("   -t     %s[REQUIRED]%s threshold to consider the transitions as 'novel'.\n", ANSI_COLOR_RED, ANSI_COLOR_RESET);
	printf("   -c     %s[REQUIRED]%s to report transitions that are not relevant as well [0 or 1].\n", ANSI_COLOR_RED, ANSI_COLOR_RESET);
	printf("   -n     %s[REQUIRED]%s number of weights to be loaded.\n", ANSI_COLOR_RED, ANSI_COLOR_RESET);
	printf("   -g     %s[REQUIRED]%s game name to be played.\n", ANSI_COLOR_RED, ANSI_COLOR_RESET);
	printf("   -h     print this help and exit\n");
	printf("\n");
}
/**
* Parses the parameters passed to the program in the command line.
*
* @param int argc number of arguments in the command line
* @param char** argv arguments in the command line
*/
void readParameters(int argc, char** argv){
	int option = 0;
	while ((option = getopt(argc, argv, "r:o:t:c:n:g:h")) != -1)
	{
		if (option == -1){
			break;
		}
		switch(option){
			case 'r': //Rom to be loaded
				romPath_param = optarg;
				break;
			case 'o': //File where the samples should be saved
				outputPath_param = optarg;
				break;
			case 't': //Threshold to start considering a transition interesting
				freqThreshold_param = atof(optarg);
				break;
			case 'c': //Threshold to start considering a transition interesting
				toReportAll_param = atoi(optarg);
				break;				
			case 'n':
				numOptions_param = atoi(optarg);
				break;
			case 'g':
				game_param = optarg;
				break;				
			case 'h': //Asking for help about the parameters
				printHelp(argv);
				exit(-1);
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

	//Check whether all required information is available in the command line:
	if(romPath_param.compare("") == 0 || outputPath_param.compare("") == 0
		|| freqThreshold_param <= 0.0 || (toReportAll_param != 1 && toReportAll_param != 0)
		|| numOptions_param < 0 || argc != NUM_MIN_ARGS + numOptions_param 
		|| game_param.compare("") == 0){
			printHelp(argv);
			exit(1);
	}

	for(int i = 0; i < numOptions_param; i++){
		optionsWgts.push_back(argv[ NUM_MIN_ARGS + i ]);
	}
}

void updateAverage(vector<bool> Fprev, vector<bool> F, int frame, int gameId){
	assert (Fprev.size() == F.size());
	assert (F.size() == NUM_BITS);
	
	bool toPrint = false;

	vector<int> tempVector(2 * NUM_BITS, 0);

	for(int i = 0; i < NUM_BITS; i++){
		if(!Fprev[i] && F[i]){ // 0->1
			frequency[i] = (frequency[i] * (frame - 1) + 1) / frame;
			tempVector[i] = 1;
			if(frame > FRAMES_TO_WAIT && frequency[i] < freqThreshold_param){
//				cout << i << " " << frequency[i] << " " << freqThreshold_param << endl;
				tempVector[i] = 2; //1 denotes flip, 2 denotes relevant flip
				toPrint = true;
			}
		} else{
			frequency[i] = (frequency[i] * (frame - 1) + 0) / frame;
		}		
		if(Fprev[i] && !F[i]){ // 1->0
			frequency[i + NUM_BITS] = (frequency[i + NUM_BITS] * (frame - 1) + 1) / frame;
			tempVector[i + NUM_BITS] = 1;
			if(frame > FRAMES_TO_WAIT && frequency[i + NUM_BITS] < freqThreshold_param){
//				cout << i << " " << frequency[i + NUM_BITS] << " " << freqThreshold_param << endl;
				tempVector[i + NUM_BITS] = 2;
				toPrint = true;
			}
		} else{
			frequency[i + NUM_BITS] = (frequency[i + NUM_BITS] * (frame - 1) + 0) / frame;
		}
	}

	ofstream myFileBits, myFileBytes;
	myFileBytes.open (outputPath_param + "_bytes.csv", ios::app);
	myFileBits.open (outputPath_param + "_bits.csv", ios::app);
	vector<int> bytesToPrint;
	if(toPrint){
		for(int i = 0; i < tempVector.size(); i++){
			if(toReportAll_param == 1 && tempVector[i] != 0){
				myFileBits << i << ",";
				bytesToPrint.push_back(int(i/8));
			}
			if(toReportAll_param == 0 && tempVector[i] == 2){
				myFileBits << i << ",";
				bytesToPrint.push_back(int(i/8));
			}			
		}
		myFileBits << endl;
		//Erase repeated elements:
		bytesToPrint.erase( unique( bytesToPrint.begin(), bytesToPrint.end() ), bytesToPrint.end() );
		for(int j = 0; j < bytesToPrint.size(); j++){
			myFileBytes << bytesToPrint[j] << ",";
		}
		myFileBytes << endl;
		bytesToPrint.clear();
	}
	myFileBytes.close();
	myFileBits.close();
}

void updateQValues(vector<int> &Features, vector<float> &QValues, int option){
	for(int a = 0; a < NUM_ACTIONS; a++){
		float sumW = 0;
		for(unsigned int i = 0; i < Features.size(); i++){
			sumW += w[option][a][Features[i]];
		}
		QValues[a] = sumW;
	}
}

int epsilonGreedy(vector<float> &QValues){
	float epsilon = 0.05;
	int randomActionTaken = 0;

	int action = Mathematics::argmax(QValues);
	//With probability epsilon: a <- random action in A(s)
	int random = rand();
	if((random % int(nearbyint(1.0/epsilon))) == 0) {
	//if((rand()%int(1.0/epsilon)) == 0){
		randomActionTaken = 1;
		action = rand() % NUM_ACTIONS;
	}
	return action;
}


int playActionUpdatingAvg(ALEInterface& ale, RAMFeatures *ram, BPROFeatures *features, int nextAction, int gameId){
	vector<bool> F(NUM_BITS, 0); //Set of active features
	vector<bool> Fprev;
	int reward = 0;

	//If the selected action was one of the primitive actions
	if(nextAction < NUM_ACTIONS){ 
		for(int i = 0; i < FRAME_SKIP; i++){
			reward += ale.act((Action) nextAction);
			frame++;
			Fprev.swap(F);
			F.clear();
			ram->getCompleteFeatureVector(ale.getScreen(), ale.getRAM(), F);
			F.pop_back();
			updateAverage(Fprev, F, frame, gameId);
		}
	}
	//If the selected action was one of the options
	else{
		int currentAction;
		vector<int> Fbpro;	                  //Set of features active
		vector<float> Q(NUM_ACTIONS, 0.0);    //Q(a) entries

//		int numTimesPlayed = 0;

		int option = nextAction - NUM_ACTIONS;
//		printf("%d: ", option);
		while(rand()%1000 > 1000 * PROB_TERMINATION && !ale.game_over()){
//			printf("*");
//			numTimesPlayed++;
			//Get state and features active on that state:		
			Fbpro.clear();
			features->getActiveFeaturesIndices(ale.getScreen(), ale.getRAM(), Fbpro);
			updateQValues(Fbpro, Q, option);       //Update Q-values for each possible action
			currentAction = epsilonGreedy(Q);
			//Take action, observe reward and next state:
			reward += ale.act((Action) currentAction);
			frame++;
			Fprev.swap(F);
			F.clear();
			ram->getCompleteFeatureVector(ale.getScreen(), ale.getRAM(), F);
			F.pop_back();
			updateAverage(Fprev, F, frame, gameId);
		}
//		cout << numTimesPlayed << endl;
//		printf("\n");
	}

	return reward;
}

int getNextAction(ALEInterface& ale, BPROFeatures *features){
	int totalNumActions = NUM_ACTIONS + numOptions_param;
	return rand()%totalNumActions;
}

void playGame(ALEInterface& ale, RAMFeatures *ram, BPROFeatures *bpro, int gameId){
	ale.reset_game();
	int reward = 0;
	while(!ale.game_over()){
		int nextAction = getNextAction(ale, bpro);
		reward += playActionUpdatingAvg(ale, ram, bpro, nextAction, gameId);
	}
	printf("%d) Final score: %d\n", gameId+1, reward);
}

void loadWeights(BPROFeatures *features){
	int numFeatures = features->getNumberOfFeatures();

	for(int i = 0; i < numOptions_param; i++){
		w.push_back(vector< vector<float> >(NUM_ACTIONS, vector<float>(numFeatures, 0.0)));
	}

	for(int i = 0; i < optionsWgts.size(); i++){
		string line;
		int nActions, nFeatures;
		int j, k;
		float value;

		std::ifstream weightsFile (optionsWgts[i].c_str());

		weightsFile >> nActions >> nFeatures;
		assert(nActions == NUM_ACTIONS);
		assert(nFeatures == numFeatures);

		while(weightsFile >> j >> k >> value){
			w[i][j][k] = value;
		}
	}
}


int main(int argc, char** argv){
	int numGames = 300;
	//Reading parameters from file defined as input in the run command:
	readParameters(argc, argv);
	
	RAMFeatures ramFeatures;
	BPROFeatures bproFeatures(game_param);

	loadWeights(&bproFeatures);
	
	ALEInterface ale(0);
	ale.setFloat("stochasticity", 0.00);
	ale.setInt("random_seed", seed_param);
	ale.setInt("max_num_frames_per_episode", 18000);	
	ale.loadROM(romPath_param.c_str());

	for(int i = 0; i < 2 * NUM_BITS; i++){
		frequency.push_back(0.0);
	}
	
	for(int seed = 1; seed < numGames + 1; seed++){
		srand(seed);
		playGame(ale, &ramFeatures, &bproFeatures, (seed-1));
	}

	return 0;
}
