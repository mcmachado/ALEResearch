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

	ale.setFloat("stochasticity", 0.00);
	ale.setInt("random_seed", param.getSeed());
	ale.setFloat("frame_skip", param.getNumStepsPerAction());
	ale.setInt("max_num_frames_per_episode", param.getEpisodeLength());

	ale.loadROM(param.getRomPath().c_str());
    std::string gameName=param.getRomPath().substr(param.getRomPath().find_last_of('/')+1);
    gameName = gameName.substr(0,gameName.find_last_of('.'));
    //std::copy(param.getRomPath().begin()+1+ param.getRomPath().find_last_of('/'),param.getRomPath().end(),gameName.begin());
    ale.setDifficulty(param.getDifficultyLevel());
    ale.setMode(param.getGameMode());
    ALEEnvironment<RAMFeatures> env(&ale,&features);

	//Instantiating the learning algorithm:
	SarsaLearner sarsaLearner(env,&param);

    auto modes = ale.getAvailableModes();
    auto diff = ale.getAvailableDifficulties();
    for(unsigned i = 0; i<16;i++){
        int count = 0;
        for(auto d : diff){
            ale.setDifficulty(d);
            for(auto m : modes){
                ale.setMode(m);
                sarsaLearner.loadWeights("svd_decomp/m_"+to_string(count)+"_"+to_string(i)+".w");
                double res = sarsaLearner.evaluatePolicy(env,40);
                cerr<<count<<" "<<i<<" "<<res<<endl;
                count++;
            }
        }
    }
    return 0;
}
