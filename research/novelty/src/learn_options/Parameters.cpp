#include <sstream>
#include "Parameters.hpp"

using namespace std;

Parameters::Parameters(int argc, char** argv){
	seed = -1;
	numOptions = 0;
	readParameters(argc, argv);
}

vector<string> Parameters::split(string str, char delimiter) {
  vector<string> internal;
  stringstream ss(str); // Turn the string into a stream.
  string tok;
  
  while(getline(ss, tok, delimiter)) {
    internal.push_back(tok);
  }
  
  return internal;
}

void Parameters::printHelp(char** argv){
	printf("Usage:    %s -s <SEED> -r <ROM> -t <EIGENVECTOR> -i <STATS_SVD> -o <OUTPUT_FILE> -n <NUM_OPTIONS> <OPTION_1> <OPTION_2> ... <OPTION_N>\n", argv[0]);
	printf("   -s     [REQUIRED] seed to be used.\n");
	printf("   -r     [REQUIRED] path to the ROM to be played by the agent.\n");
	printf("   -t     [REQUIRED] path to file containing the eigenvector that will generate the reward.\n");
	printf("   -i     [REQUIRED] prefix path to the files containing the mean and var. of the data (from SVD).\n");
	printf("   -o     [REQUIRED] path to the file to be written with the learned weights (an option).\n");
	printf("   -n     [REQUIRED] number of weights to be loaded.\n");
	printf("   -h     print this help and exit\n");
	printf("\n");
}

void Parameters::readParameters(int argc, char** argv){

	int option = 0;
	while ((option = getopt(argc, argv, "s:r:t:i:o:n:h")) != -1)
	{
		if (option == -1){
			break;
		}
		switch(option){
			case 's':
				seed = atoi(optarg);
				break;
			case 'r':
				romPath = optarg;
				break;
			case 't':
				eigVectorPath = optarg;
				break;
			case 'i':
				statEigVectorPath = optarg;
				break;
			case 'o':
				outputPath = optarg;
				break;
			case 'n':
				numOptions = atoi(optarg);
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

	//Check whether all required information is available in the command line:
	if(romPath.compare("") == 0 || outputPath.compare("") == 0
		|| eigVectorPath.compare("") == 0 || numOptions < 0 
		|| statEigVectorPath.compare("") == 0 || seed < 0
		|| argc != NUM_MIN_ARGS + numOptions){
			printHelp(argv);
			exit(1);
	}
	vector<string> splitPath  = split(romPath, '.');
	vector<string> splitPath2 = split(splitPath[splitPath.size()-2], '/');
	gameName = splitPath2[splitPath2.size()-1];

	for(int i = 0; i < numOptions; i++){
		optionsWgts.push_back(argv[ NUM_MIN_ARGS + i ]);
	}
}
