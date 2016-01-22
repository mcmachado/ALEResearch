#include "common/Parameters.hpp"
#include "agents/rl/sarsa/SarsaLearner.hpp"
#include "agents/rl/sarsa_split/SarsaSplitLearner.hpp"
#include "environments/river_swim/RiverSwimEnvironment.hpp"
#include "features/RiverSwimFeatures.hpp"

#define NUM_SEEDS 30

int main(int argc, char** argv){
	//Reading parameters from file defined as input in the run command:
	Parameters param(argc, argv);
	int firstSeed = param.getSeed();
	for(int i = 0; i < NUM_SEEDS; i++){
		int seed = firstSeed + i;
		srand(seed);
		
		RiverSwimFeatures features;
	    RiverSwimEnvironment<RiverSwimFeatures> env(&features);

		//Instantiating the learning algorithm:
		SarsaSplitLearner sarsaLearner(env, &param, seed * 2);
	    //Learn a policy:
	    sarsaLearner.learnPolicy(env);
	}
    return 0;
}
