/****************************************************************************************
** This class reads parameters from a file informed in the command line and store the
** relevant information. It is ideal to centralize all parameters here, as well as defining
** most of them in a file, to ease the execution of several experiments.
** 
** USEFUL INFORMATION: If one wants to add a new parameter, all he needs to do is to put
**   this parameter in the appropriate .cfg file and then add such parameter in the last
**   lines of the method parseParametersFromConfigFile. You also need to add the
**   parameter in the class Parameters, defining its get and set methods.
**
** Author: Marlos C. Machado
***************************************************************************************/

#ifndef GRAPHICS_H
#define GRAPHICS_H
#include "../../../../../src/common/Graphics.hpp"
#endif
#ifndef PARAMETERS_H
#define PARAMETERS_H
#include "ParametersLoadingWeights.hpp"
#endif
#include <getopt.h>
#include <string>
#include <map>
#include <fstream>
#include <vector>
#include <assert.h>
#include <sstream>
#include <stdlib.h>

void Parameters::printHelp(char** argv){
	printf("Usage:    %s-s <SEED> -c <CONF_FILE> -r <ROM> -t <EIGEN_VECTOR> -i <EIGEN_VECTOR_STATS> -w <FREQ> -n 3 <EIG1> <EIG2> <EIG3>\n", argv[0]);
	printf("   -s     %s[REQUIRED]%s seed to random number generator.\n", ANSI_COLOR_RED, ANSI_COLOR_RESET);
	printf("   -c     %s[REQUIRED]%s path to file with configuration info.\n", ANSI_COLOR_RED, ANSI_COLOR_RESET);
	printf("   -r     %s[REQUIRED]%s path to the rom to be played by the agent.\n", ANSI_COLOR_RED, ANSI_COLOR_RESET);
	printf("   -t     %s[REQUIRED]%s path to file that contains the eigenvector description (for the reward).\n", ANSI_COLOR_RED, ANSI_COLOR_RESET);
	printf("   -i     %s[REQUIRED]%s prefix path to files that contain mean and var. of the data (from SVD).\n", ANSI_COLOR_RED, ANSI_COLOR_RESET);
	printf("   -n     %s[REQUIRED]%s The number of weights (representing options) have to be loaded. If 0, no weight file is loaded.\n", ANSI_COLOR_RED, ANSI_COLOR_RESET);
	printf("   -w     If one wants to save intermediate weights, this is prefix to files that will store the agent's learned weights every FREQUENCY_SAVING episodes.\n");
	printf("   -h     print this help and exit\n");
	printf("\n");
}

Parameters::Parameters(int argc, char** argv){

	this->setOptionRewardPath("");

	this->readParameters(argc, argv);
	//Get the game being played by the path to ROM:
	size_t pos = 0;
	std::string token;
	std::string delimiter = "/";
	std::string toParse = this->getRomPath();
	while ((pos = toParse.find(delimiter)) != std::string::npos) {
    	token = toParse.substr(0, pos);
    	toParse.erase(0, pos + delimiter.length());
	}
	//The game is what was left in toParse, now I get the first part, without extension:
	this->gameBeingPlayed = toParse.substr(0, toParse.find("."));
	this->parseParametersFromConfigFile(this->getConfigPath());

	if(this->getOptionRewardPath().compare("") == 0){
		printHelp(argv);
		exit(1);
	}

	if(this->getNumOptionsLoad() > 0){
		for(int i = 0; i < this->getNumOptionsLoad(); i++){
			int idx = argc - i - 1;
			pathToOptionFiles.push_back(argv[idx]);
		}
	}
}

std::vector<std::string> Parameters::parseLine(std::string line){
	std::vector<std::string> pair; //Store pair <ID, Value>
	//First remove all white spaces
	for (int i = line.length(); i > 0; --i) {
	    if(line[i] == ' '){
	    	line.erase(i, 1);
	    }
	}
	//Check if it is a comment
	if(line.length() > 0 && line[0]!='#'){
		//If it is not a comment, parse line
		std::string delimiter = "=";
		size_t pos = 0;
		std::string token;
		//Store pair in the vector of parameters
		while((pos = line.find(delimiter)) != std::string::npos){
    		token = line.substr(0, pos);
    		pair.push_back(token);
    		line.erase(0, pos + delimiter.length());
		}
		pair.push_back(line);
	}
	return pair;
}

void Parameters::readParameters(int argc, char* argv[]){
	int option = 0;
	while ((option = getopt(argc, argv, "c:r:s:t:w:i:n:h")) != -1)
	{
		if (option == -1){
			break;
		}
		switch(option)
		{
			case 'c':
				this->setConfigPath(optarg);
				break;
			case 'h':
				printHelp(argv);
				exit(-1);
			case 'r':
				this->setRomPath(optarg);
				break;
			case 't':
				this->setOptionRewardPath(optarg);
				break;
			case 's':
				this->setSeed(optarg);
				break;
			case 'w':
				this->setFileWithWeights(optarg);
				this->setToSaveWeightsAfterLearning(1);
				break;
			case 'i':
				this->setDataStatsPath(optarg);
				break;
			case 'n':
				this->setNumOptionsLoad(atoi(optarg));
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
	if(this->getRomPath().compare("") == 0 || this->getConfigPath().compare("") == 0 || this->getSeed() == 0
		|| (this->getToSaveWeightsAfterLearning() && this->getDataStatsPath().compare("") == 0) 
		|| (argc != 13 + this->getNumOptionsLoad() && argc != 15 + this->getNumOptionsLoad())){
		printHelp(argv);
		exit(-1);
	}
}

void Parameters::parseParametersFromConfigFile(std::string cfgFileName){
	std::string line;
	//Open config file passed as parameter
	std::ifstream cfgFile(cfgFileName.c_str());
	//Save parameters temporarily in a Map to ease its retrieval later
	std::map<std::string,std::string> parameters;
	if (cfgFile.is_open()){
		//Read file line by line:
		while( getline(cfgFile, line)){
			if(line.length() > 0){
				std::vector<std::string> parsed = parseLine(line);
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

	this->setAlpha(atof(parameters["ALPHA"].c_str()));
	this->setGamma(atof(parameters["GAMMA"].c_str()));
	this->setEpsilon(atof(parameters["EPSILON"].c_str()));
	this->setLambda(atof(parameters["LAMBDA"].c_str()));
	this->setDisplay(atoi(parameters["DISPLAY"].c_str()));
	this->setEpisodeLength(atoi(parameters["EPISODE_LENGTH"].c_str()));
	this->setNumEpisodesLearn(atoi(parameters["NUM_EPISODES_LEARN"].c_str()));
	this->setNumEpisodesEval(atoi(parameters["NUM_EPISODES_EVAL"].c_str()));
	this->setNumStepsPerAction(atoi(parameters["NUM_STEPS_PER_ACTION"].c_str()));
	this->setNumRows(atoi(parameters["NUM_ROWS"].c_str()));
	this->setNumColumns(atoi(parameters["NUM_COLUMNS"].c_str()));
	this->setNumColors(atoi(parameters["NUM_COLORS"].c_str()));
	this->setIsMinimalAction(atoi(parameters["USE_MIN_ACTIONS"].c_str()));
	this->setTraceThreshold(atof(parameters["TRACE_THRESHOLD"].c_str()));
	this->setUseRewardSign(atoi(parameters["USE_REWARD_SIGN"].c_str()));
	this->setSubtractBackground(atoi(parameters["SUBTRACT_BACKGROUND"].c_str()));
	this->setOptimisticInitialization(atoi(parameters["OPTIMISTIC_INIT"].c_str()));

	this->setFrequencySavingWeights(atoi(parameters["FREQUENCY_SAVING"].c_str()));
	this->setToLoadWeights(atoi(parameters["LOAD_WEIGHTS"].c_str()));
	this->setPathToWeightsFiles(parameters["WEIGHTS_TO_LOAD"]);
	this->setLearningLength(atoi(parameters["TOTAL_FRAMES_LEARN"].c_str()));

	if(this->getSubtractBackground()){
		std::string folderWithBackgrounds = parameters["PATH_TO_BACKGROUND"];
		setPathToBackground(folderWithBackgrounds, this->gameBeingPlayed);
	}
}

void Parameters::setOptionRewardPath(std::string name){
	this->optionRewardPath = name;
}

std::string Parameters::getOptionRewardPath(){
	return this->optionRewardPath;
}

void Parameters::setPathToBackground(std::string path, std::string romFile){
	pathToBackground = path + romFile + std::string(".bg");
}

std::string Parameters::getPathToBackground(){
	return this->pathToBackground;
}

void Parameters::setRomPath(std::string name){
	this->romPath = name;
}

std::string Parameters::getRomPath(){
	return this->romPath;
}

void Parameters::setConfigPath(std::string name){
	this->configPath = name;
}

std::string Parameters::getConfigPath(){
	return this->configPath;
}

int Parameters::getSeed(){
	return this->seed;
}

void Parameters::setSeed(std::string name){
	this->seed = atoi(name.c_str());
}

void Parameters::setAlpha(float a){
	this->alpha = a;
}

float Parameters::getAlpha(){
	return this->alpha;
}

void Parameters::setGamma(float a){
	this->gamma = a;
}

float Parameters::getGamma(){
	return this->gamma;
}

void Parameters::setEpsilon(float a){
	this->epsilon = a;
}

float Parameters::getEpsilon(){
	return this->epsilon;
}

void Parameters::setLambda(float a){
	this->lambda = a;
}

float Parameters::getLambda(){
	return this->lambda;
}

void Parameters::setDisplay(int a){
	this->display = a;
}

int Parameters::getDisplay(){
	return this->display;
}

void Parameters::setNumEpisodesLearn(int a){
	this->numEpisodesLearn = a;
}

void Parameters::setNumEpisodesEval(int a){
	this->numEpisodesEval = a;
}

int Parameters::getNumEpisodesLearn(){
	return this->numEpisodesLearn;
}

int Parameters::getNumEpisodesEval(){
	return this->numEpisodesEval;
}

void Parameters::setEpisodeLength(int a){
	this->episodeLength = a;
}

int Parameters::getNumStepsPerAction(){
	return this->numStepsPerAction;
}

void Parameters::setNumStepsPerAction(int a){
	this->numStepsPerAction = a;
}

int Parameters::getEpisodeLength(){
	return this->episodeLength;
}

void Parameters::setNumRows(int a){
	this->numRows = a;
}

int Parameters::getNumRows(){
	return this->numRows;
}

void Parameters::setNumColumns(int a){
	this->numColumns = a;
}

int Parameters::getNumColumns(){
	return this->numColumns;
}

void Parameters::setNumColors(int a){
	this->numColors = a;
}

int Parameters::getNumColors(){
	return this->numColors;
}

void Parameters::setIsMinimalAction(int a){
	this->minimalAction = a;
}

int Parameters::isMinimalAction(){
	return this->minimalAction;
}

float Parameters::getTraceThreshold(){
	return this->traceThreshold;
}

void Parameters::setTraceThreshold(float a){
	this->traceThreshold = a;
}

void Parameters::setUseRewardSign(int a){
	this->useRewardSign = a;
}

int Parameters::getUseRewardSign(){
	return this->useRewardSign;
}

void Parameters::setSubtractBackground(int a){
	this->subtractBackground = a;
}

int Parameters::getSubtractBackground(){
	return this->subtractBackground;
}

void Parameters::setOptimisticInitialization(int a){
	this->toUseOptimisticInit = a;
}

int Parameters::getOptimisticInitialization(){
	return this->toUseOptimisticInit;
}

void Parameters::setToSaveWeightsAfterLearning(int a){
	this->toSaveWeightsAfterLearning = a;
}

int Parameters::getToSaveWeightsAfterLearning(){
	return this->toSaveWeightsAfterLearning;
}

void Parameters::setFileWithWeights(std::string name){
	this->fileWithWeights = name;
}

std::string Parameters::getFileWithWeights(){
	return this->fileWithWeights;
}

int Parameters::getFrequencySavingWeights(){
	return this->frequencySavingWeights;
}

void Parameters::setFrequencySavingWeights(int a){
	this->frequencySavingWeights = a;
}

int Parameters::getToLoadWeights(){
	return this->toLoadWeights;
}

void Parameters::setToLoadWeights(int a){
	this->toLoadWeights = a;
}

std::string Parameters::getPathToWeightsFiles(){
	return this->pathToWeightsFiles;
}

void Parameters::setPathToWeightsFiles(std::string name){
	this->pathToWeightsFiles = name;
}

int Parameters::getLearningLength(){
	return this->learningLength;
}

void Parameters::setLearningLength(int a){
	this->learningLength = a;
}

void Parameters::setDataStatsPath(std::string name){
	this->pathToDataStatsPath = name;
}

std::string Parameters::getDataStatsPath(){
	return this->pathToDataStatsPath;
}

void Parameters::setNumOptionsLoad(int a){
	numOptionsLoad = a;
}

int Parameters::getNumOptionsLoad(){
	return numOptionsLoad;
}