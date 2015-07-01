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

#include "input/Parameters.hpp"
#include "svd/DimReduction.hpp"
#include "control/ControlAgent.hpp"

int main(int argc, char** argv){
	Parameters param(argc, argv);

	int maxNumIterations = 4;

	for(int i = 0; i < maxNumIterations; i++){
		gatherSamplesFromRandomTrajectories();
		reduceDimensionalityOfEvents();
		learnOptionsDerivedFromEigenEvents();
	}
}