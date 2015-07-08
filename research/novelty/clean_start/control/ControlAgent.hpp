/* Author: Marlos C. Machado */
#include <ale_interface.hpp>

#include "../control/Agent.hpp"
#include "../common/Parameters.hpp"

void learnOptionsDerivedFromEigenEvents();

int epsilonGreedy(Agent &agent, vector<float> &QValues, float epsilon);

void updateAverage(Parameters *param, Agent &agent, vector<bool> Fprev, vector<bool> F, int frame, int iter);

void updateQValues(Agent &agent, vector<int> &Features, vector<float> &QValues, int option);

void gatherSamplesFromRandomTrajectories(ALEInterface& ale, Parameters *param, Agent &agent, int iter);

int playGame(ALEInterface& ale, Parameters *param, int iter);

int playActionUpdatingAvg(ALEInterface& ale, Parameters *param, int &frame, int nextAction, int iter);