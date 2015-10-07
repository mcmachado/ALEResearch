#include "common/Parameters.hpp"
#include "agents/rl/sarsa/SarsaLearner.hpp"
#include "environments/mountain_car/MountainCarEnvironment.hpp"
#include "features/MountainCarFeatures.hpp"

double curIter;

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

using namespace std;
int main(int argc, char** argv){
	//Reading parameters from file defined as input in the run command:
	Parameters param(argc, argv);
	srand(param.getSeed());
	
	MountainCarFeatures features;
	//Reporting parameters read:
	printBasicInfo(param);
	
    MountainCarEnvironment<MountainCarFeatures> env(&features);
    //float scores[5];
    float scores;
    //for(int i = 0; i < 5; i++){
    //    cout<<endl<<"FLAVOR "<<i<<endl;
    //    env.setFlavor(i);
    env.setFlavor(0);
        //Instantiating the learning algorithm:
        SarsaLearner sarsaLearner(env, &param, param.getSeed());
        //Learn a policy:
        sarsaLearner.learnPolicy(env);
        //sarsaLearner.showGreedyPol();
        printf("\n\n== Evaluation without Learning == \n\n");
        //scores[i] = sarsaLearner.evaluatePolicy(env);
        scores = sarsaLearner.evaluatePolicy(env);
    //}
    //cout<<"Final Scores "<<endl;
    
    return 0;
}