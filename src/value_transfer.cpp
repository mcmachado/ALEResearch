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
    /*ale.setDifficulty(param.getDifficultyLevel());
      ale.setMode(param.getGameMode());*/
    ALEEnvironment<BasicFeatures> env(&ale,&features);

	//Instantiating the learning algorithm:
	SarsaLearner sarsaLearner(env,&param);
    //Learn a policy:
    //sarsaLearner.learnPolicy(env);
    //sarsaLearner.saveWeightsToFile("weights_"+gameName+"_RAM_d"+std::to_string(param.getDifficultyLevel())+"_m"+std::to_string(param.getGameMode())+".w");
    auto modes = ale.getAvailableModes();
    auto diff = ale.getAvailableDifficulties();

    sarsaLearner.loadWeights("results/weights/weights_"+gameName+"_BASIC_d"+std::to_string(diff[0])+"_m"+std::to_string(modes[0])+".w");
    
    printf("\n\n== Evaluation reference == \n\n");
    double ref = sarsaLearner.evaluatePolicy(env);
    cerr<<"Reference score is "<<ref<<endl;
    double tot_var=0;
    int tot=0;
    for(const auto& m : modes){
        ale.setMode(m);
        for(const auto& d : diff){
            ale.setDifficulty(d);
            if(d==diff[0]&&modes[0]==m){
                continue;
            }
            printf("\n\n== Evaluation mode %d difficulty %d == \n\n",m,d);
            double score=sarsaLearner.evaluatePolicy(env);
            cerr<<"mode "<<m<<" difficulty "<<d<<" score "<<score<<" variation "<<100*fabs(ref-score)/score<<"%"<<endl;
            tot_var+=score;
            tot+=1;
        }
    }
    cerr<<"mean variation "<<tot_var/(double)tot<<endl;
    return 0;
}
