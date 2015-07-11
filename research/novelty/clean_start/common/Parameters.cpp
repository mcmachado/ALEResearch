/* Author: Marlos C. Machado */

#include "Parameters.hpp"

#define ANSI_COLOR_RED     "\x1b[31m"
#define ANSI_COLOR_RESET   "\x1b[0m"

#include <getopt.h>
#include <string>
#include <map>
#include <fstream>
#include <vector>
#include <assert.h>
#include <sstream>
#include <stdlib.h>

void Parameters::printHelp(char** argv){
	printf("Usage:    %s -s <SEED> -c <CONF_FILE> -r <ROM>\n", argv[0]);
	printf("   -s     %s[REQUIRED]%s seed to random number generator.\n", ANSI_COLOR_RED, ANSI_COLOR_RESET);
	printf("   -c     %s[REQUIRED]%s path to file with configuration info.\n", ANSI_COLOR_RED, ANSI_COLOR_RESET);
	printf("   -r     %s[REQUIRED]%s path to the rom to be played by the agent.\n", ANSI_COLOR_RED, ANSI_COLOR_RESET);
	printf("   -h     print this help and exit\n");
	printf("\n");
}

Parameters::Parameters(int argc, char** argv){

	readParameters(argc, argv);

	//Get the game being played by the path to ROM:
	size_t pos = 0;
	string token;
	string delimiter = "/";
	string toParse = romPath;
	while ((pos = toParse.find(delimiter)) != string::npos) {
    	token = toParse.substr(0, pos);
    	toParse.erase(0, pos + delimiter.length());
	}

	//The game is what was left in toParse, now I get the first part, without extension:
	gameBeingPlayed = toParse.substr(0, toParse.find("."));
	parseParametersFromConfigFile(configPath);
}

void Parameters::readParameters(int argc, char** argv){
	int option = 0;
	while ((option = getopt(argc, argv, "c:r:s:h")) != -1)
	{
		if (option == -1){
			break;
		}
		switch(option)
		{
			case 'c':
				configPath = optarg;
				break;
			case 'r':
				romPath = optarg;
				break;
			case 's':
				seed = atoi(optarg);
				break;
			case 'h':
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
	//Check if all parameters were properly set, otherwise interrupt	
	if(romPath.compare("") == 0 || configPath.compare("") == 0 || seed == 0){
		printHelp(argv);
		exit(-1);
	}
}

void Parameters::parseParametersFromConfigFile(string cfgFileName){
	string line;
	//Open config file passed as parameter
	ifstream cfgFile(cfgFileName.c_str());
	//Save parameters temporarily in a Map to ease its retrieval later
	map<string, string> parameters;
	if (cfgFile.is_open()){
		//Read file line by line:
		while(getline(cfgFile, line)){
			if(line.length() > 0){
				vector<string> parsed = parseLine(line);
				//Add the parsed elements to an appropriate structure:
				if(parsed.size() == 2){
					parameters[parsed[0]] = parsed[1];
				}
			}
		}
		cfgFile.close();
	}
	else{
		printf("Unable to open the file '%s', defined as the configuration file.\n", cfgFileName.c_str());
	}

	//Algorithm parameters:
	alpha                 = atof(parameters["ALPHA"                ].c_str());
	gamma                 = atof(parameters["GAMMA"                ].c_str());
	epsilon               = atof(parameters["EPSILON"              ].c_str());
	lambda                = atof(parameters["LAMBDA"               ].c_str());
	traceThreshold        = atof(parameters["TRACE_THRESHOLD"      ].c_str());
	rarityFreqThreshold   = atof(parameters["RARITY_FREQ_THRESHOLD"].c_str());
	optionTerminationProb = atof(parameters["PROB_TERM_OPTION"     ].c_str());

	//Execution parameters:
	display                    = atoi(parameters["DISPLAY"                    ].c_str());
	maxNumIterations           = atoi(parameters["NUM_ITERATIONS"             ].c_str());
	episodeLength              = atoi(parameters["EPISODE_LENGTH"             ].c_str());
	learningLength             = atoi(parameters["TOTAL_FRAMES_LEARN"         ].c_str());
	isMinimalAction            = atoi(parameters["USE_MIN_ACTIONS"            ].c_str());
	framesToDefRarity          = atoi(parameters["NUM_FRAMES_DEF_RARITY"      ].c_str());
	numStepsPerAction          = atoi(parameters["NUM_STEPS_PER_ACTION"       ].c_str());
	numNewOptionsPerIter       = atoi(parameters["NUM_OPTIONS_PER_ITER"       ].c_str());
    frequencySavingWeights     = atoi(parameters["FREQUENCY_SAVING"           ].c_str());
    numGamesToSampleRareEvents = atoi(parameters["GAMES_TO_SAMPLE_RARE_EVENTS"].c_str());


	//Feature set parameters:
	numRows            = atoi(parameters["NUM_ROWS"           ].c_str());
	numColors          = atoi(parameters["NUM_COLORS"         ].c_str());
	numColumns         = atoi(parameters["NUM_COLUMNS"        ].c_str());
	subtractBackground = atoi(parameters["SUBTRACT_BACKGROUND"].c_str());

	if(subtractBackground){
		string folderWithBackgrounds = parameters["PATH_TO_BACKGROUND"];
		pathToBackground = folderWithBackgrounds + gameBeingPlayed + std::string(".bg");
	}
}

vector<string> Parameters::parseLine(string line){
	vector<string> pair; //Store pair <ID, Value>
	//First remove all white spaces
	for (int i = line.length(); i > 0; --i) {
	    if(line[i] == ' '){
	    	line.erase(i, 1);
	    }
	}
	//Check if it is a comment
	if(line.length() > 0 && line[0]!='#'){
		//If it is not a comment, parse line
		string delimiter = "=";
		size_t pos = 0;
		string token;
		//Store pair in the vector of parameters
		while((pos = line.find(delimiter)) != string::npos){
    		token = line.substr(0, pos);
    		pair.push_back(token);
    		line.erase(0, pos + delimiter.length());
		}
		pair.push_back(line);
	}
	return pair;
}
