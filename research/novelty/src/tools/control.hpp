#ifndef CONTROL_HPP
#define CONTROL_HPP

#include <ale_interface.hpp>
#include "../features/BPROFeatures.hpp"

int epsilonGreedy(std::vector<float> &QValues, int numTotalActions);

void updateQValues(std::vector<int> &Features, std::vector<float> &QValues, 
	std::vector<std::vector<float> > &w);

int playOption(ALEInterface& ale, BPROFeatures features, int option, 
	ActionVect actions, std::vector<std::vector<std::vector<float> > > &primitiveOptions);

int takeAction(ALEInterface& ale, BPROFeatures features, int actionToTake,
	ActionVect actions, std::vector<std::vector<std::vector<float> > > &primitiveOptions);

#endif 
