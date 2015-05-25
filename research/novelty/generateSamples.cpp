/*******************************************************************************
*** This file is supposed to implement an algorithm that plays random keeping **
*** track of the frequency of the features it has seen. When a feature        **
*** transition becomes rare (defined via threshold parameter), and the agent  **
*** sees it, it prints the indicator function for that coordinate.            **
***                                                                           **
*** TODO: Right now it is implemented to just use RAM features. If one wants  **
*** to use screen-based features, this has to be implemented, because some    **
*** additional information (e.g. num tiles, backgrounds) should be provided.  **
*******************************************************************************/

#include <getopt.h>
#include <vector>

#include <ale_interface.hpp>

#include "../../src/common/Graphics.hpp"
#include "../../src/features/RAMFeatures.hpp"

//Features:
#define NUM_BYTE 128
#define NUM_BITS 1024
//Agent's:
#define NUM_ACTS 18
#define FRAME_SKIP 5
//Colours:
#define KRED  "\x1B[31m"
#define KNRM  "\x1B[0m"
//Average:
#define FRAMES_TO_WAIT 300

double freqThreshold = 0.0;

string romPath;
string outputPath;

vector<double> frequency;         //[0:1023] transitions 0->1; [1024:2048] transitions 1->0
int toReportAll = -1;

/**
* Prints the instructions related to the program's parameters in a friendly mode
*
* @param char** argv received from the command line
*/
void printHelp(char** argv){
	printf("Usage:    %s./generateSamples -r <ROM> -f <FREQ_FILE> -t <ACTIONS_FILE> -n <FREQ_THRESHOLD> -o <OUTPUT_FILE_PREFIX> \n", argv[0]);
	printf("   -r     %s[REQUIRED]%s path to the rom to be played by the agent.\n", ANSI_COLOR_RED, ANSI_COLOR_RESET);
	printf("   -o     %s[REQUIRED]%s path to the file to be written with the output.\n", ANSI_COLOR_RED, ANSI_COLOR_RESET);
	printf("   -t     %s[REQUIRED]%s threshold to consider the transitions as 'novel'.\n", ANSI_COLOR_RED, ANSI_COLOR_RESET);
	printf("   -c     %s[REQUIRED]%s to report transitions that are not relevant as well [0 or 1].\n", ANSI_COLOR_RED, ANSI_COLOR_RESET);
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
	while ((option = getopt(argc, argv, "r:o:t:c:h")) != -1)
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
		|| freqThreshold <= 0.0 || (toReportAll != 1 && toReportAll != 0)){
			printHelp(argv);
			exit(1);
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

void playGame(ALEInterface& ale, Features *features, int gameId){
	vector<bool> F(NUM_BITS, 0); //Set of active features
	vector<bool> Fprev;
	ale.reset_game();

	int frame = 0;
	int reward = 0;
	while(!ale.game_over()){
		int nextAction = rand() % NUM_ACTS;
		for(int i = 0; i < FRAME_SKIP; i++){
			reward += ale.act((Action) nextAction);
			frame++;
			Fprev.swap(F);
			F.clear();
			features->getCompleteFeatureVector(ale.getScreen(), ale.getRAM(), F);
			F.pop_back();
			updateAverage(Fprev, F, frame, gameId);
		}
	}
	printf("%d) Final score: %d\n", gameId+1, reward);
}


int main(int argc, char** argv){
	int numGames = 300;
	readParameters(argc, argv);

	ALEInterface ale(0);
	RAMFeatures features;
	ale.setInt("random_seed", 1);
	ale.setFloat("stochasticity", 0.00);
	ale.loadROM(romPath.c_str());

	for(int i = 0; i < 2 * NUM_BITS; i++){
		frequency.push_back(0.0);
	}

	//int seed = atoi(argv[7]);
	for(int seed = 1; seed < numGames + 1; seed++){
		srand(seed);
		playGame(ale, &features, (seed-1));
	}

	return 0;
}
