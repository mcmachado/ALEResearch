#ifndef PARAMETERS_HPP
#define PARAMETERS_HPP

#include <string>
#include <vector>
#include <getopt.h>

#define NUM_MIN_ARGS 13

class Parameters{
	public:
		//Parameters:
		int seed, numOptions;
		std::vector<std::string> optionsWgts;

		std::string fileToLoadWeights;
		std::string romPath, gameName, outputPath;
		std::string statEigVectorPath, eigVectorPath;

		Parameters(int argc, char** argv);

	private:
		Parameters();
		void printHelp(char** argv);
		void readParameters(int argc, char** argv);
		std::vector<std::string> split(std::string str, char delimiter);

};
#endif
