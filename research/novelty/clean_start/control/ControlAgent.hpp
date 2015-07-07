/* Author: Marlos C. Machado */
#include <ale_interface.hpp>

#include "../control/Agent.hpp"
#include "../common/Parameters.hpp"
#include "../observations/RAMFeatures.hpp"
#include "../observations/BPROFeatures.hpp"

void learnOptionsDerivedFromEigenEvents();

int epsilonGreedy(Agent &agent, vector<float> &QValues, float epsilon);

void updateAverage(vector<bool> Fprev, vector<bool> F, int frame);

void updateQValues(Agent &agent, vector<int> &Features, vector<float> &QValues, int option);

void gatherSamplesFromRandomTrajectories(ALEInterface& ale, Parameters *param, Agent &agent);

int playGame(ALEInterface& ale, Parameters *param, RAMFeatures *ramFeatures, BPROFeatures *bproFeatures);

int playActionUpdatingAvg(ALEInterface& ale, Parameters *param, 
	RAMFeatures *ramFeatures, BPROFeatures *bproFeatures, int &frame, int nextAction);