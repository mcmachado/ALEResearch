/* Author: Marlos C. Machado */

#include <ale_interface.hpp>

#include "../control/Agent.hpp"
#include "../common/Parameters.hpp"


//Gathering data to obtain eigenvectors:
int playGame(ALEInterface& ale, Parameters *param, int iter, vector<vector<bool> > &dataset);

void gatherSamplesFromRandomTrajectories(ALEInterface& ale, Parameters *param, Agent &agent,
	vector<vector<bool> > &dataset, int iter);

//Learning Options:
int argmax(std::vector<float> array);

void learnOptions(ALEInterface &ale, Parameters *param, Agent &agent, RAMFeatures ramFeatures,
	BPROFeatures bproFeatures, int option);

void cleanTraces(vector<vector<float> > &e, vector<vector<int> > &nonZeroElig);

void learnOptionsDerivedFromEigenEvents(ALEInterface &ale, Parameters *param,
	Agent &agent, std::vector<float> &datasetMeans, std::vector<float> &datasetStds,
	std::vector<std::vector<float> > &eigenVectors, vector<vector<vector<float> > > learnedOptions, int iter);
