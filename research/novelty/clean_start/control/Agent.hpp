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
		int numberOfAvailActions, numberOfOptions, numberOfPrimitiveActions;

		vector<vector<float> > e;       //Eligibility trace
		vector<vector<int> >nonZeroElig; //To optimize the implementation
		/* This is the theta vector, the one containing the weights that really try
		to maximize the external rewards (score). They are learned only at the end.
		What we really learn during each iteration are the options, which the learned
		weights are not w, but learnedOptions*/
		std::vector<std::vector<std::vector<float> > > w;
		/* This matrix is initially empty, but each new learned option is going to add its
		learned set of weights on it. Therefore, when initialized it will be 0x0. After learnined
		the first set of options (e.g. 5) it will be 5 x |A| x |F|, where |A| is 18 and |F| is the
		size of the feature vector. In the second iteration it be concatenated to another matrix
		with dimensions 5 x |A + 5| x |F|. */
		vector<vector<vector<float> > > learnedOptions;

		Agent(ALEInterface& ale, Parameters *param);
		int playActionUpdatingAvg(ALEInterface& ale, Parameters *param, int &frame, 
			int nextAction, int iter, vector<vector<bool> > &dataset);

		void cleanTraces();
		void sanityCheck(vector<float> &QValues);
		int epsilonGreedy(vector<float> &QValues, float epsilon);
		void updateReplTrace(Parameters *param, int action, vector<int> &Features);
		void updateQValues(vector<vector<float> > &learnedOptions, vector<int> &Features, vector<float> &QValues, int option);
	private:
		Agent();
		void updateAverage(Parameters *param, vector<bool> Fprev, vector<bool> F, 
			int frame, int iter, vector<vector<bool> > &dataset);
};

#endif