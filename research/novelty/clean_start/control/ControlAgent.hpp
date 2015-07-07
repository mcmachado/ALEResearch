/* Author: Marlos C. Machado */
#include <ale_interface.hpp>

#include "../input/Parameters.hpp"
#include "../observations/RAMFeatures.hpp"
#include "../observations/BPROFeatures.hpp"

int playGame(ALEInterface& ale, RAMFeatures *ram, BPROFeatures *features);

void gatherSamplesFromRandomTrajectories(ALEInterface& ale, Parameters *param);

void learnOptionsDerivedFromEigenEvents();