#include "vector"
#include "constants.hpp"

#include <ale_interface.hpp>

int epsilonGreedy(std::vector<float> &QValues);
int getNextAction(ALEInterface& ale, int numOptions);
void updateQValues(std::vector<int> &Features, std::vector<float> &QValues, 
	std::vector<std::vector<std::vector<float> > > &w, int option);