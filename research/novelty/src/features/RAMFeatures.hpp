/****************************************************************************************
*** Implementation of RAM Features, described in details in the paper below.           **
***       "The Arcade Learning Environment: An Evaluation Platform for General Agents. **
***        Marc G. Bellemare, Yavar Naddaf, Joel Veness, and Michael Bowling.          **
***        Journal of Artificial Intelligence Research, 47:253–279, 2013."             **
***                																	   **
*** The idea is to get the RAM state and each bit be a feature in the feature vector.  **
***																					   **
*** Author: Marlos C. Machado														   **
*****************************************************************************************/

#ifndef RAM_FEATURES_H
#define RAM_FEATURES_H

#include <ale_interface.hpp>

class RAMFeatures{
	private:
	public:
		RAMFeatures();
		~RAMFeatures();
		int getNumberOfFeatures();

		void getActiveFeaturesIndices(const ALERAM &ram, std::vector<int>& features);	
		void getCompleteFeatureVector(const ALERAM &ram, vector<bool>& features);
};

#endif
