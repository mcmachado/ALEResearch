#include <ale_interface.hpp>
#include "common/Parameters.hpp"
#include "agents/rl/qlearning/QLearner.hpp"
#include "agents/rl/sarsa/SarsaLearner.hpp"
#include "agents/rl/true_online_sarsa/TrueOnlineSarsaLearner.hpp"
#include "agents/baseline/ConstantAgent.hpp"
#include "agents/baseline/PerturbAgent.hpp"
#include "agents/baseline/RandomAgent.hpp"
#include "agents/human/HumanAgent.hpp"
#include "features/RAMFeatures.hpp"
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
    RAMFeatures features(&param);
	//Reporting parameters read:
	printBasicInfo(param);
	
	ALEInterface ale(param.getDisplay());

	ale.setFloat("repeat_action_probability", 0.00);
	ale.setInt("random_seed", param.getSeed());
	ale.setFloat("frame_skip", param.getNumStepsPerAction());
	ale.setInt("max_num_frames_per_episode", param.getEpisodeLength());

	ale.loadROM(param.getRomPath().c_str());
    std::string gameName=param.getRomPath().substr(param.getRomPath().find_last_of('/')+1);
    gameName = gameName.substr(0,gameName.find_last_of('.'));
    ALEEnvironment<RAMFeatures> env(&ale,&features);
    auto modes = ale.getAvailableModes();
    auto diff = ale.getAvailableDifficulties();
    // ale.setDifficulty(diff[0]);
    // ale.setMode(modes[0]);
    //ale.setDifficulty(param.getDifficultyLevel());
    //ale.setMode(param.getGameMode());
    ale.setDifficulty(1);
    ale.setMode(3);

	//Instantiating the learning algorithm:
	SarsaLearner sarsaLearner(env,&param);
    //Learn a policy:
    cout<<diff[0]<<" "<<modes[0]<<endl;
    cout<<"results/weights/weights_"+gameName+"_RAM_d"+std::to_string(diff[0])+"_m"+std::to_string(modes[0])+".w"<<endl;
    //sarsaLearner.loadWeights("results/weights/weights_"+gameName+"_BASIC_d"+std::to_string(diff[0])+"_m"+std::to_string(modes[0])+".w");
    //sarsaLearner.loadWeights("results_VTR/VTR_freeway_BASIC_d0_m0/relearnt_weights_freeway_BASIC_d0_m0.w");
    std::vector<Action> act;
    if(param.isMinimalAction()){
        act = env.getMinimalActionSet();
    }else{
        act = env.getLegalActionSet();
    }

    int numFeatures = env.getNumberOfFeatures();
    int numActions = act.size();
    vector<vector<float> > onpolicy_weights(numActions,std::vector<float>(numFeatures,0.0));
    ifstream f("results/weights/weights_"+gameName+"_RAM_d"+std::to_string(diff[0])+"_m"+std::to_string(modes[0])+".w");
    int a, b;
    f>>a>>b;
    assert(a==numActions && b==numFeatures);
    int i,j;
    float value;
    while( f >> i >> j >> value){
        onpolicy_weights[i][j] = value;
    }


    sarsaLearner.learnPolicy(env,true,onpolicy_weights);
    //sarsaLearner.saveWeightsToFile("weights_"+gameName+"_BASIC_d"+std::to_string(param.getDifficultyLevel())+"_m"+std::to_string(param.getGameMode())+".w");
    printf("\n\n== Evaluation without Learning == \n\n");
    double res = sarsaLearner.evaluatePolicy(env);
    cerr<<"Obtained score is "<<res<<endl;
    sarsaLearner.learnPolicy(env);
    printf("\n\n== Evaluation without Learning == \n\n");
    res = sarsaLearner.evaluatePolicy(env);
    cerr<<"Obtained score is "<<res<<endl;

    return 0;
}
