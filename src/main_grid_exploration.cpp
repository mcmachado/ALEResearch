#include "common/Parameters.hpp"
#include "agents/rl/sarsa/SarsaLearner.hpp"
#include "agents/rl/sarsa_split/SarsaSplitLearner.hpp"
#include "environments/grid/GridEnvironment.hpp"
#include "features/GridFeatures.hpp"



void printBasicInfo(Parameters param){
	printf("Seed: %d\n", param.getSeed());
	printf("\nCommand Line Arguments:\nPath to Config. File: %s\nPath to ROM File: %s\nPath to Backg. File: %s\n", 
		param.getConfigPath().c_str(), param.getRomPath().c_str(), param.getPathToBackground().c_str());
	if(param.getSubtractBackground()){
		printf("\nBackground will be subtracted...\n");
	}
	printf("\nParameters read from Configuration File:\n");
	printf("alpha:   %f\ngamma:   %f\nepsilon: %f\nlambda:  %f\nep. length: %d\n\n", 
		param.getAlpha(), param.getGamma(), param.getEpsilon(), param.getLambda(), 
		param.getEpisodeLength());
}


int main(int argc, char** argv){
	//Reading parameters from file defined as input in the run command:
	Parameters param(argc, argv);
	srand(param.getSeed());
	
	BasicGridFeatures features;
	//Reporting parameters read:
	printBasicInfo(param);
	
    GridEnvironment<BasicGridFeatures> env(&features);
    env.setFlavor(1);

    SarsaSplitLearner sarsaSplitLearner(env, &param, param.getSeed() * 2);
    sarsaSplitLearner.learnPolicy(env);

	SarsaLearner sarsaLearner(env, &param, param.getSeed() * 2);
    sarsaLearner.learnPolicy(env);
    
    //sarsaLearner.showGreedyPol();
    
    return 0;
}
