/* Author: Marlos C. Machado */

#include <string>
#include <vector>

using namespace std;

class Parameters{
	private:
		/**
 		* Parse parameters read from the command line.
 		*/
		void readParameters(int argc, char** argv);
		/**
 		* Print what is the list of parameters to be used.
 		*/
		void printHelp(char** argv);
		/**
 		* Parse parameters from the configurations file. Once it is defined we extract each
 		* of the relevant parameters, storing them to future use.
 		*/
		void parseParametersFromConfigFile(string cfgFileName);
		/**
 		* Parse a line from the configuration file returning the pair <ID, Value>.
 		*/
		vector<string> parseLine(string line);

	public:
		//I decided to not use gets and sets here.
		float alpha;                    //step-size
		float gamma;                    //discount factor
		float epsilon;                  //exploration probability
		float lambda;                   //trace
		float traceThreshold;           //threshold to make the trace zero, to avoid very small values

		int seed;                       //seed to be used by the random number generator
		int display;                    //if it should display screen
		int numIterations;              //number of times I am going to iterate over all steps in my algorithm
		int episodeLength;              //length of a single episode
		int learningLength;             //The number of frames to be learned, in total
		int isMinimalAction;            //use only valid actions for the game or all the Atari legal actions
		int numStepsPerAction;          //number of frames the agent perfoms similarly to speed-up the game
		int frequencySavingWeights;     //If we are asked to save the weights, we need to know how frequently (in frames)

		int numRows;                    //number of rows for feature representation
		int numColumns;                 //number of columns for feature representation
		int numColors;                  //colors to be considered in the feature representation of the screen
		int subtractBackground;         //whether the background should be removed when generating screen-based features
		string pathToBackground;        //path to the file containing the game background

		string romPath;                 //rom to be executed, informed in the command line
		string configPath;              //path to the file with all the other parameters, informed in the command line
		string gameBeingPlayed;         //name of the game being played, this is used internally

		/**
		* Constructor
		*/
		Parameters(int argc, char** argv);

};