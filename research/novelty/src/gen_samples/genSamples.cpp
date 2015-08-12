/*******************************************************************************
*** This implements an algorithm that plays randomly keeping track of the     **
*** frequency of the features it has seen. When a feature transition becomes  **
*** rare (defined via threshold parameter), and the agent sees it, it prints  **
*** the indicator function for that coordinate. It can load weights           **
*** (representing options) as well. If weights are loaded they are treated    **
*** as primitive actions that will be used by the random agent as well.       **
*******************************************************************************/

#include <getopt.h>
#include <vector>

#include <ale_interface.hpp>

#include "../features/BPROFeatures.hpp"
#include "../features/RAMFeatures.hpp"
#include "../common/Mathematics.hpp"

#define NUM_ACTIONS      18
#define NUM_MIN_ARGS     15
#define MAX_NUM_FRAMES   2500000
#define PROB_TERMINATION 0.01


//Features:
#define NUM_BYTE 128
#define NUM_BITS 1024
//Agent's:
#define NUM_ACTS 18
#define FRAME_SKIP 5
//Average:
#define FRAMES_TO_WAIT 300

double freqThreshold = 0.0;

vector<double> frequency;         //[0:1023] transitions 0->1; [1024:2048] transitions 1->0

//Parameters:
int seed        =  1;
int numOptions  =  0;
int toReportAll = -1;

string romPath;
string gameName;
string outputPath;
string game_param;

vector<vector<vector<float> > > w;

vector<string> split(string str, char delimiter) {
  vector<string> internal;
  stringstream ss(str); // Turn the string into a stream.
  string tok;
  
  while(getline(ss, tok, delimiter)) {
    internal.push_back(tok);
  }
  
  return internal;
}

void printHelp(char** argv){
	printf("Usage:    %s -s <SEED> -r <ROM> -f <FREQ_FILE> -t <ACTIONS_FILE> -n <FREQ_THRESHOLD> -o <OUTPUT_FILE_PREFIX> -c <REPORT_IRR_TRANS> -n <NUM_OPTIONS> <OPTION_1> <OPTION_2> ... <OPTION_N>\n", argv[0]);
	printf("   -s     [REQUIRED] seed to be used.\n");
	printf("   -r     [REQUIRED] path to the rom to be played by the agent.\n");
	printf("   -o     [REQUIRED] path to the file to be written with the output.\n");
	printf("   -t     [REQUIRED] threshold to consider the transitions as 'novel'.\n");
	printf("   -c     [REQUIRED] to report transitions that are not relevant as well [0 or 1].\n");
	printf("   -n     [REQUIRED] number of weights to be loaded.\n");
	printf("   -h     print this help and exit\n");
	printf("\n");
}

void readParameters(int argc, char** argv){
	int option = 0;
	while ((option = getopt(argc, argv, "s:r:o:t:c:n:h")) != -1)
	{
		if (option == -1){
			break;
		}
		switch(option){
			case 'r': //Rom to be loaded
				romPath = optarg;
				break;
			case 'o': //File where the samples should be saved
				outputPath = optarg;
				break;
			case 't': //Threshold to start considering a transition interesting
				freqThreshold = atof(optarg);
				break;
			case 'c': //Threshold to start considering a transition interesting
				toReportAll = atoi(optarg);
				break;
			case 'n':
				numOptions = atoi(optarg);
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
	if(romPath.compare("") == 0 || outputPath.compare("") == 0
		|| freqThreshold <= 0.0 || (toReportAll != 1 && toReportAll != 0)
		|| numOptions < 0 || argc != NUM_MIN_ARGS + numOptions){
			printHelp(argv);
			exit(1);
	}
	vector<string> splitPath  = split(romPath, '.');
	vector<string> splitPath2 = split(splitPath[splitPath.size()-2], '/');
	gameName = splitPath2[splitPath2.size()-1];
}

int epsilonGreedy(vector<float> &QValues){
	float epsilon = 0.05;
	int randomActionTaken = 0;

	int action = Mathematics::argmax(QValues);
	//With probability epsilon: a <- random action in A(s)
	int random = rand();
	if((random % int(nearbyint(1.0/epsilon))) == 0) {
		randomActionTaken = 1;
		action = rand() % NUM_ACTIONS;
	}
	return action;
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

void updateAverage(vector<bool> Fprev, vector<bool> F, int frame, int gameId){
	assert (Fprev.size() == F.size());
	assert (F.size() == NUM_BITS);
	
	bool toPrint = false;

	vector<int> tempVector(2 * NUM_BITS, 0);

	for(int i = 0; i < NUM_BITS; i++){
		if(!Fprev[i] && F[i]){ // 0->1
			frequency[i] = (frequency[i] * (frame - 1) + 1) / frame;
			tempVector[i] = 1;
			if(frame > FRAMES_TO_WAIT && frequency[i] < freqThreshold){
				tempVector[i] = 2; //1 denotes flip, 2 denotes relevant flip
				toPrint = true;
			}
		} else{
			frequency[i] = (frequency[i] * (frame - 1) + 0) / frame;
		}		
		if(Fprev[i] && !F[i]){ // 1->0
			frequency[i + NUM_BITS] = (frequency[i + NUM_BITS] * (frame - 1) + 1) / frame;
			tempVector[i + NUM_BITS] = 1;
			if(frame > FRAMES_TO_WAIT && frequency[i + NUM_BITS] < freqThreshold){
				tempVector[i + NUM_BITS] = 2;
				toPrint = true;
			}
		} else{
			frequency[i + NUM_BITS] = (frequency[i + NUM_BITS] * (frame - 1) + 0) / frame;
		}
	}

	ofstream myFileBits, myFileBytes;
	myFileBytes.open (outputPath + "_bytes.csv", ios::app);
	myFileBits.open (outputPath + "_bits.csv", ios::app);
	vector<int> bytesToPrint;
	if(toPrint){
		for(int i = 0; i < tempVector.size(); i++){
			if(toReportAll == 1 && tempVector[i] != 0){
				myFileBits << i << ",";
				bytesToPrint.push_back(int(i/8));
			}
			if(toReportAll == 0 && tempVector[i] == 2){
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

int playActionUpdatingAvg(ALEInterface& ale, RAMFeatures *ram, BPROFeatures *features, int nextAction, int gameId){
	vector<bool> F(NUM_BITS, 0); //Set of active features
	vector<bool> Fprev;
	int reward = 0;

	//If the selected action was one of the primitive actions
	if(nextAction < NUM_ACTIONS){ 
		for(int i = 0; i < FRAME_SKIP; i++){
			reward += ale.act((Action) nextAction);
			Fprev.swap(F);
			F.clear();
			ram->getCompleteFeatureVector(ale.getRAM(), F);
			F.pop_back();
			updateAverage(Fprev, F, ale.getEpisodeFrameNumber(), gameId);
		}
	}
	//If the selected action was one of the options
	else{
		int currentAction;
		vector<int> Fbpro;	                  //Set of features active
		vector<float> Q(NUM_ACTIONS, 0.0);    //Q(a) entries

		int option = nextAction - NUM_ACTIONS;
		while(rand()%1000 > 1000 * PROB_TERMINATION && !ale.game_over()){
			//Get state and features active on that state:		
			Fbpro.clear();
			features->getActiveFeaturesIndices(ale.getScreen(), Fbpro);
			updateQValues(Fbpro, Q, option);       //Update Q-values for each possible action
			currentAction = epsilonGreedy(Q);
			//Take action, observe reward and next state:
			reward += ale.act((Action) currentAction);
			Fprev.swap(F);
			F.clear();
			ram->getCompleteFeatureVector(ale.getRAM(), F);
			F.pop_back();
			updateAverage(Fprev, F, ale.getEpisodeFrameNumber(), gameId);
		}
	}

	return reward;
}

int getNextAction(ALEInterface& ale){
	int totalNumActions = NUM_ACTIONS + numOptions;
	return rand()%totalNumActions;
}

void playGame(ALEInterface& ale, RAMFeatures *ram, BPROFeatures *bpro, int gameId){
	int score = 0;
	while(!ale.game_over()){
		int nextAction = getNextAction(ale);
		score += playActionUpdatingAvg(ale, ram, bpro, nextAction, gameId);
	}
	printf("Episode: %d, Final score: %d\n", gameId+1, score);
}

int main(int argc, char** argv){
	readParameters(argc, argv);
	srand(seed);

	RAMFeatures ramFeatures;
	BPROFeatures bproFeatures(gameName);

	ALEInterface ale(0);
	ale.setInt  ("random_seed"               , seed);
	ale.setInt  ("max_num_frames_per_episode", 18000);
	ale.setBool ("color_averaging"           , true );
	ale.setFloat("frame_skip"                , 5    );
	ale.setFloat("repeat_action_probability" , 0.00 );

	ale.loadROM(romPath.c_str());

	for(int i = 0; i < 2 * NUM_BITS; i++){
		frequency.push_back(0.0);
	}

	int totalNumFrames = 0;
	while(totalNumFrames <= MAX_NUM_FRAMES){
		srand(seed);
		playGame(ale, &ramFeatures, &bproFeatures, (seed-1));

		ale.reset_game();
		totalNumFrames += ale.getEpisodeFrameNumber();
	}

	return 0;
}
