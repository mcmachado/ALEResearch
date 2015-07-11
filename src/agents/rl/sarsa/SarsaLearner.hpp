/****************************************************************************************
** Implementation of Sarsa(lambda). It implements Fig. 8.8 (Linear, gradient-descent 
** Sarsa(lambda)) from the book "R. Sutton and A. Barto; Reinforcement Learning: An 
** Introduction. 1st edition. 1988."
** Some updates are made to make it more efficient, as not iterating over all features.
**
** TODO: Make it as efficient as possible.
** 
** Author: Marlos C. Machado
***************************************************************************************/

#ifndef SARSALEARNER_H
#define SARSALEARNER_H
#include "../RLLearner.hpp"
#include <vector>
#include <unordered_map>
#include <cstdio>
#include <cmath>

class SarsaLearner : public RLLearner<bool>{
	public:
		float alpha, delta, lambda, traceThreshold;
		int numFeatures, currentAction, nextAction;
		int toSaveWeightsAfterLearning, saveWeightsEveryXSteps;

		std::string nameWeightsFile, pathWeightsFileToLoad;

		std::vector<int> F;					//Set of features active
		std::vector<int> Fnext;              //Set of features active in next state
		std::vector<float> Q;               //Q(a) entries
		std::vector<float> Qnext;           //Q(a) entries for next action
                std::vector<std::unordered_map<int,float> > e;      //Eligibility trace
		std::vector<std::vector<float> > w;      //Theta, weights vector

		/**
 		* Constructor declared as private to force the user to instantiate SarsaLearner
 		* informing the parameters to learning/execution.
 		*/
		SarsaLearner();
		/**
 		* This method evaluates whether the Q-values are sound. By unsound I mean huge Q-values (> 10e7)
 		* or NaN values. If so, it finishes the execution informing the algorithm has diverged. 
 		*/
		void sanityCheck();
		/**
 		* In Sarsa the Q-values (one per action) are updated as the sum of weights for that given action.
 		* To avoid writing it more than once on the code, its update was extracted to a separate function.
 		* It updates the vector<float> Q assuming that vector<int> F is filled, as it sums just the weights
 		* that are active in F.
 		*/
    void updateQValues(std::vector<int> &Features, std::vector<float> &QValues);
		/**
 		* When using Replacing traces, all values not related to the current action are set to 0, while the
 		* values for the current action that their features are active are set to 1. The traces decay following
 		* the rule: e[action][i] = gamma * lambda * e[action][i]. It is possible to also define thresholding.
 		*/
		void updateReplTrace(int action, std::vector<int> &Features);
		/**
 		* When using Replacing traces, all values not related to the current action are set to 0, while the
 		* values for the current action that their features are active are added 1. The traces decay following
 		* the rule: e[action][i] = gamma * lambda * e[action][i]. It is possible to also define thresholding.
 		*/
		void updateAcumTrace(int action, std::vector<int> &Features);
	public:
		/**
 		* Prints the weights in a file. Each line will contain a weight.
 		*/
                void saveWeightsToFile(std::string suffix="");
		/**
 		* Loads the weights saved in a file. Each line will contain a weight.
 		*/		
		void loadWeights();
        void loadWeights(std::string fname);
    SarsaLearner(Environment<bool>& env, Parameters *param);
		/**
 		* Implementation of an agent controller. This implementation is Sarsa(lambda).
 		*
 		* @param Environment& env Arcade Learning Environment interface: object used to define agents'
 		*        actions, obtain simulator's screen, RAM, etc.
 		*/
		void learnPolicy(Environment<bool>& env);
		/**
 		* After the policy was learned it is necessary to evaluate its quality. Therefore, a given number
 		* of episodes is run without learning (the vector of weights and the trace are not updated).
 		*
 		* @param Environment& env Arcade Learning Environment interface: object used to define agents'
 		*        actions, obtain simulator's screen, RAM, etc.
 		*/
		double evaluatePolicy(Environment<bool>& env);
		/**
		* Destructor, not necessary in this class.
		*/
		~SarsaLearner();
};


#endif
