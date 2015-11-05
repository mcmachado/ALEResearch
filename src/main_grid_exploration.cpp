#include "common/Parameters.hpp"
#include "agents/rl/sarsa/SarsaLearner.hpp"
#include "agents/rl/sarsa_split/SarsaSplitLearner.hpp"
#include "environments/grid/GridEnvironment.hpp"
#include "features/GridFeatures.hpp"

int main(int argc, char** argv){
	//Reading parameters from file defined as input in the run command:
	Parameters param(argc, argv);
	srand(param.getSeed());
	
	BasicGridFeatures features;
	
    GridEnvironment<BasicGridFeatures> env(&features);
    env.setFlavor(1);

//    printf("Opt. Sarsa: ");
    SarsaSplitLearner sarsaSplitLearner(env, &param, param.getSeed() * 2);
    sarsaSplitLearner.learnPolicy(env);

//    srand(param.getSeed());
//    printf("Sarsa: ");
//    GridEnvironment<BasicGridFeatures> env2(&features);
//    env2.setFlavor(1);

//	SarsaLearner sarsaLearner(env2, &param, param.getSeed() * 2);
//    sarsaLearner.learnPolicy(env2);
    
    return 0;
}
