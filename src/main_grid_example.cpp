#include "common/Parameters.hpp"
#include "agents/rl/qlearning/QLearner.hpp"
#include "agents/rl/sarsa/SarsaLearner.hpp"
#include "agents/rl/true_online_sarsa/TrueOnlineSarsaLearner.hpp"
#include "agents/baseline/ConstantAgent.hpp"
#include "agents/baseline/PerturbAgent.hpp"
#include "agents/baseline/RandomAgent.hpp"
#include "agents/human/HumanAgent.hpp"
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

	//Instantiating the learning algorithm:
	SarsaLearner sarsaLearner(env, &param, param.getSeed() * 2);
    //Learn a policy:
    sarsaLearner.learnPolicy(env);
    //sarsaLearner.showGreedyPol();
    printf("\n\n== Evaluation without Learning == \n\n");
    std::vector<Action> act;
    if(param.isMinimalAction()){
        act = env.getMinimalActionSet();
    }else{
        act = env.getLegalActionSet();
    }
    
    return 0;
}
