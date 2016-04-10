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
		int numberOfEigenBehaviours, numberOfPrimitiveActions;

		vector<float> transitions;

		/* This is the theta vector, the one containing the weights that really try
		to maximize the external rewards (score). They are learned only at the end.
		What we really learn during each iteration are the eigenbehaviours, which the
		learned weights are not w, but learnedEigenbehaviours*/
		std::vector<std::vector<std::vector<float> > > w;
		/* This matrix is initially empty, but each new learned eigenbehaviour is going to add its
		learned set of weights on it. Therefore, when initialized it will be 0x0. After learnined
		the first set of eigenbehaviours (e.g. 5) it will be 5 x |A| x |F|, where |A| is 18 and |F|
		is the size of the feature vector. In the second iteration it be concatenated to another
		matrix with dimensions 5 x |A + 5| x |F|. */
		vector<vector<vector<float> > > learnedEigenBehaviours;

		Agent(ALEInterface& ale, Parameters *param);

		void act(ALEInterface& ale, int action, Parameters *param, std::vector<float> &mean,
			std::vector<float> &std, std::vector<float> &eigenVectors, vector<float> &reward);

		int actStoringObservation(ALEInterface& ale, Parameters *param, int &frame, 
			int nextAction, int iter, vector<vector<char> > &dataset);

		int getNumAvailActions();
		void sanityCheck(vector<float> &QValues);
		int epsilonGreedy(vector<float> &QValues, float epsilon);
		void updateTransitionVector(vector<bool> F, vector<bool> Fnext);
		void cleanTraces(vector<vector<float> > &e, vector<vector<int> > &nonZeroElig);
		void updateQValues(vector<vector<float> > &learnedEigenBehaviours, vector<int> &Features, vector<float> &QValues, int eigenbehaviour);
		void updateReplTrace(Parameters *param, int action, vector<int> &Features, 
			vector<vector<float> > &e, vector<vector<int> > &nonZeroElig);
	private:
		Agent();
		void storeDifferenceBetweenObservations(Parameters *param, vector<bool> Fprev, vector<bool> F, 
			int frame, int iter, vector<vector<char> > &dataset);

		int playOption(ALEInterface& ale, float epsilon, int eigenbehaviour);
};

#endif