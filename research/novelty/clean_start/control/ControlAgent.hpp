/* Author: Marlos C. Machado */
#include <ale_interface.hpp>

#include "../control/Agent.hpp"
#include "../common/Parameters.hpp"

void learnOptionsDerivedFromEigenEvents();

void gatherSamplesFromRandomTrajectories(ALEInterface& ale, Parameters *param, Agent &agent, int iter);

int playGame(ALEInterface& ale, Parameters *param, int iter);