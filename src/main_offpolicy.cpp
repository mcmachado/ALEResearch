/****************************************************************************************
** Starting point for running Sarsa algorithm. Here the parameters are set, the algorithm
** is started, as well as the features used. In fact, in order to create a new learning
** algorithm, once its class is implementend, the main file just need to instantiate
** Parameters, the Learner and the type of Features to be used. This file is a good 
** example of how to do it. A parameters file example can be seen in ../conf/sarsa.cfg.
** This is an example for other people to use: Sarsa with Basic Features.
** 
** Author: Marlos C. Machado
***************************************************************************************/

#include <ale_interface.hpp>
#include "common/Parameters.hpp"
#include "agents/rl/qlearning/QLearner.hpp"
#include "agents/rl/sarsa/SarsaLearner.hpp"
#include "agents/rl/true_online_sarsa/TrueOnlineSarsaLearner.hpp"
#include "agents/baseline/ConstantAgent.hpp"
#include "agents/baseline/PerturbAgent.hpp"
#include "agents/baseline/RandomAgent.hpp"
#include "agents/human/HumanAgent.hpp"
#include "features/BasicFeatures.hpp"
#include "environments/ale/ALEEnvironment.hpp"
#include "offPolicy/GQLearner.hpp"

using namespace std;
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
	
	//Using Basic features:
	BasicFeatures features(&param);
	//Reporting parameters read:
	printBasicInfo(param);
	
	ALEInterface ale(param.getDisplay());

	ale.setFloat("stochasticity", 0.00);
	ale.setInt("random_seed", param.getSeed());
	ale.setFloat("frame_skip", param.getNumStepsPerAction());
	ale.setInt("max_num_frames_per_episode", param.getEpisodeLength());

	ale.loadROM(param.getRomPath().c_str());
    std::string gameName=param.getRomPath().substr(param.getRomPath().find_last_of('/')+1);
    gameName = gameName.substr(0,gameName.find_last_of('.'));
    ALEEnvironment<BasicFeatures> env(&ale,&features);
    auto modes = ale.getAvailableModes();
    auto diff = ale.getAvailableDifficulties();
    // ale.setDifficulty(diff[0]);
    // ale.setMode(modes[0]);
    ale.setDifficulty(param.getDifficultyLevel());
    ale.setMode(param.getGameMode());
    //ale.setDifficulty(0);
    //ale.setMode(5);

	//Instantiating the learning algorithm:
	SarsaLearner sarsaLearner(env,&param);
    //Learn a policy:
    cout<<diff[0]<<" "<<modes[0]<<endl;
    cout<<"results/weights/weights_"+gameName+"_BASIC_d"+std::to_string(diff[0])+"_m"+std::to_string(modes[0])+".w"<<endl;
    sarsaLearner.loadWeights("results/weights/weights_"+gameName+"_BASIC_d"+std::to_string(diff[0])+"_m"+std::to_string(modes[0])+".w");
    //sarsaLearner.loadWeights("results_VTR/VTR_freeway_BASIC_d0_m0/relearnt_weights_freeway_BASIC_d0_m0.w");
    std::vector<Action> act;
    if(param.isMinimalAction()){
        act = env.getMinimalActionSet();
    }else{
        act = env.getLegalActionSet();
    }
    std::shared_ptr<OffPolicyLearner> off(new GQLearner(env.getNumberOfFeatures(),act,&param));
    env.setOffPolicyLearner(off);
    //printf("\n\n== Evaluation reference == \n\n");
    double ref = sarsaLearner.evaluatePolicy(env,10);
    cerr<<"Reference score is "<<ref<<endl;
    //copy weights
    sarsaLearner.w=std::dynamic_pointer_cast<GQLearner>(off)->weights;
    //disable offpolicy
    env.setOffPolicyLearner(nullptr);
    
    //ReLearn a policy:
    sarsaLearner.learnPolicy(env);
    sarsaLearner.saveWeightsToFile("weights_"+gameName+"_BASIC_d"+std::to_string(param.getDifficultyLevel())+"_m"+std::to_string(param.getGameMode())+".w");
    printf("\n\n== Evaluation without Learning == \n\n");
    ref = sarsaLearner.evaluatePolicy(env);
    cerr<<"Obtained score is "<<ref<<endl;

    return 0;
}
