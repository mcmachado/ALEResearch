/* Author: Marlos C. Machado */

#include <ale_interface.hpp>

#include "../control/Agent.hpp"
#include "../common/Parameters.hpp"


//Gathering data to obtain eigenvectors:
int playGame(ALEInterface& ale, Parameters *param, int iter, vector<vector<char> > &dataset);

void gatherSamplesFromRandomTrajectories(ALEInterface& ale, Parameters *param, Agent &agent,
	vector<vector<char> > &dataset, int iter);

//Learning Eigenbehaviours:
void learnEigenBehaviours(ALEInterface &ale, Parameters *param, Agent &agent,
	std::vector<float> &datasetMeans, std::vector<float> &datasetStds,
	std::vector<float> &eigenVectors, int eigenbehaviour);

void cleanTraces(vector<vector<float> > &e, vector<vector<int> > &nonZeroElig);

void learnEigenBehavioursDerivedFromEigenPurposes(ALEInterface &ale, Parameters *param,
	Agent &agent, std::vector<float> &datasetMeans, std::vector<float> &datasetStds,
	std::vector<std::vector<float> > &eigenVectors, int iter);
