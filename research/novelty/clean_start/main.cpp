/****************************************************************************************
** Starting point for running my algorithm. In this code, the agent first plays some   **
** random trajectories keeping track of the frequency of each feature. Then he defines **
** what are interesting events (feature flips) that should be added to the table that  **
** will have its dimensionality reduced. Once the dimensionality is reduced, each of   **
** the obtained eigen-vectors are used to generate a reward function that will be used **
** to learn options. These options will then be added to the agent's action set and    **
** the process will start over.                                                        **
**                                                                                     **
** Author: Marlos C. Machado                                                           **
*****************************************************************************************/

#include <ale_interface.hpp>

#include "control/Agent.hpp"
#include "svd/DimReduction.hpp"
#include "common/Parameters.hpp"
#include "control/ControlAgent.hpp"

using namespace std;

void initializeALE(ALEInterface &ale, const Parameters param){
	ale.setInt  ("random_seed"               , param.seed);
	ale.setInt  ("max_num_frames_per_episode", param.episodeLength);
	ale.setBool ("sound"                     , param.display);
	ale.setBool ("display_screen"            , param.display);
	ale.setFloat("stochasticity"             , 0.00);
	/*In a first moment I am not using this because I need to see the
	  intermediate screens. If I use this I will just see every 5th
	  screen, if a bit flips and unflips between these 5 frames I will
	  not see it. I don't think it is a good idea. I may revisit this
	  assumption later, testing it.*/
	//ale.setFloat("frame_skip"                , param.numStepsPerAction);
	
	ale.loadROM(param.romPath.c_str());
}

int main(int argc, char** argv){

	Parameters param(argc, argv);
	srand(param.seed);

	int maxNumIterations = param.numIterations;
	
	ALEInterface ale;
	initializeALE(ale, param);

	//TODO: decide if I am going to define the feature set here,
	//in the agent, or in the control.
	Agent agent(ale, &param);

	for(int iter = 0; iter < maxNumIterations; iter++){
		gatherSamplesFromRandomTrajectories(ale, &param, agent, iter);
		reduceDimensionalityOfEvents();
		learnOptionsDerivedFromEigenEvents();
	}

	return 1;
}
