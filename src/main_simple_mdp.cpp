#include "common/Parameters.hpp"
#include "agents/rl/sarsa/SarsaLearner.hpp"
#include "agents/rl/sarsa_split/SarsaSplitLearner.hpp"
#include "environments/simple_mdps/4_state.hpp"
#include "features/TabularRepresentation.hpp"

int main(int argc, char** argv){
	//Reading parameters from file defined as input in the run command:
	Parameters param(argc, argv);
	int seed = param.getSeed();
	srand(seed);
		
	TabularRepresentation features;
	FourStatesEnvironment<TabularRepresentation> env(&features);

	//Instantiating the learning algorithm:
	SarsaSplitLearner sarsaLearner(env, &param, seed * 2);
	//Learn a policy:
	sarsaLearner.learnPolicy(env);

    return 0;
}
