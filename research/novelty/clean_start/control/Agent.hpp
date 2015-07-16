/* Author: Marlos C. Machado */

#include <ale_interface.hpp>

#include "../common/Parameters.hpp"
#include "../observations/RAMFeatures.hpp"
#include "../observations/BPROFeatures.hpp"

#ifndef AGENT_H
#define AGENT_H

#include <vector>

class Agent{
	public:
		RAMFeatures  ramFeatures;
		BPROFeatures bproFeatures;
		ActionVect actions; //Basic actions
		std::vector<float> freqOfBitFlips; //[0:1023] transitions 0->1; [1024:2048] transitions 1->0
		std::vector<std::vector<std::vector<float> > > w;  //Theta, weights vector
		int numberOfAvailActions, numberOfOptions, numberOfPrimitiveActions;

		Agent(ALEInterface& ale, Parameters *param);
		int playActionUpdatingAvg(ALEInterface& ale, Parameters *param, int &frame, 
			int nextAction, int iter, vector<vector<bool> > &dataset);

		int epsilonGreedy(vector<float> &QValues, float epsilon);
		void updateQValues(vector<int> &Features, vector<float> &QValues, int option);
	private:
		Agent();
		void updateAverage(Parameters *param, vector<bool> Fprev, vector<bool> F, 
			int frame, int iter, vector<vector<bool> > &dataset);
};

#endif