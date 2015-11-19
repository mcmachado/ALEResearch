#include "common/Parameters.hpp"
#include "agents/rl/fastSarsa/FastSarsaLearner.hpp"
#include "environments/mountain_car/MountainCarEnvironment.hpp"
#include "features/MountainCarFeatures.hpp"
#include "common/Timer.hpp"

using namespace std;
int main(int argc, char** argv){
    struct timeval tvBegin, tvEnd, tvDiff;
    gettimeofday(&tvBegin, NULL);
	//Reading parameters from file defined as input in the run command:
	Parameters param(argc, argv);
	srand(param.getSeed());
	
	MountainCarFeatures features;
	
    MountainCarEnvironment<MountainCarFeatures> env(&features);
    float scores;
    env.setFlavor(0);
    
    //Instantiating the learning algorithm:
    FastSarsaLearner fastSarsa(env, &param, param.getSeed());
    //Learn a policy:
    fastSarsa.learnPolicy(env);
    //printf("\n\n== Evaluation without Learning == \n\n");
    //scores = fastSarsa.evaluatePolicy(env);
    gettimeofday(&tvEnd, NULL);
    timeval_subtract(&tvDiff, &tvEnd, &tvBegin);
    float elapsedTime = float(tvDiff.tv_sec) + float(tvDiff.tv_usec)/1000000.0;
    printf("%fs\n", elapsedTime);

    return 0;
}