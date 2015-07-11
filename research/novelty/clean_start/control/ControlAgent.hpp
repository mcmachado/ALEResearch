/* Author: Marlos C. Machado */

#include <ale_interface.hpp>

#include "../control/Agent.hpp"
#include "../common/Parameters.hpp"

void learnOptionsDerivedFromEigenEvents();

void gatherSamplesFromRandomTrajectories(ALEInterface& ale, Parameters *param, Agent &agent, int iter, vector<vector<bool> > &dataset);

int playGame(ALEInterface& ale, Parameters *param, int iter, vector<vector<bool> > &dataset);