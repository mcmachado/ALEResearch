#include "common/Parameters.hpp"
#include "agents/rl/sarsa/SarsaLearner.hpp"
#include "agents/rl/sarsa_split/SarsaSplitLearner.hpp"
#include "environments/river_swim/RiverSwimEnvironment.hpp"
#include "features/RiverSwimFeatures.hpp"

int main(int argc, char** argv){
	//Reading parameters from file defined as input in the run command:
	Parameters param(argc, argv);
	srand(param.getSeed());
	
	RiverSwimFeatures features;
    RiverSwimEnvironment<RiverSwimFeatures> env(&features);

	//Instantiating the learning algorithm:
	SarsaSplitLearner sarsaLearner(env, &param, param.getSeed() * 2);
    //Learn a policy:
    sarsaLearner.learnPolicy(env);

    return 0;
}
