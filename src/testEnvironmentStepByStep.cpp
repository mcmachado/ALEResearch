#include "common/Parameters.hpp"
#include "environments/mountain_car/MountainCarEnvironment.hpp"
#include "features/MountainCarFeatures.hpp"

double curIter;

using namespace std;
int main(int argc, char** argv){
	//Reading parameters from file defined as input in the run command:
	Parameters param(argc, argv);
	srand(param.getSeed());
	
	MountainCarFeatures features;
	
    MountainCarEnvironment<MountainCarFeatures> env(&features);
    env.setFlavor(0);
    
    ActionVect actions;
    if(param.isMinimalAction()){
		actions = env.getMinimalActionSet();
	} else{
		actions = env.getLegalActionSet();
	}

	vector<int> F;
	env.getActiveFeaturesIndices(F);
    for(int i = 0; i < F.size(); i++){
    	cout << F[i] << " ";
    }
    std::cout << std::endl;

    while(!env.isTerminal()){
    	int action;
    	std::cin >> action;
    	env.act(actions[action]);
    	F.clear();
    	env.getActiveFeaturesIndices(F);

    	for(int i = 0; i < F.size(); i++){
    		cout << F[i] << " ";
    	}
    	std::cout << std::endl;
    }
    
    return 0;
}