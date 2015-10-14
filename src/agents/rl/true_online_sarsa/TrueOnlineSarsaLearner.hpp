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

#ifndef TRSARSALEARNER_H
#define TRSARSALEARNER_H

#include "../RLLearner.hpp"
#include <vector>

template<typename FeatureType>
class TrueOnlineSarsaLearner : public RLLearner<FeatureType>{
	private:
		float alpha, delta, lambda, traceThreshold;
		int numFeatures, currentAction, nextAction;

		std::string nameWeightsFile;

		std::vector<int> F;					//Set of features active
		std::vector<int> Fnext;              //Set of features active in next state
		std::vector<float> Q;               //Q(a) entries
		std::vector<float> Qnext;           //Q(a) entries for next action
		std::vector<std::vector<float> > e;      //Eligibility trace
		std::vector<std::vector<float> > w;      //Theta, weights vector
		std::vector<std::vector<int> >nonZeroElig;//To optimize the implementation 

		/**
 		* Constructor declared as private to force the user to instantiate TrueOnlineSarsaLearner
 		* informing the parameters to learning/execution.
 		*/
		TrueOnlineSarsaLearner();
		/**
 		* This method evaluates whether the Q-values are sound. By unsound I mean huge Q-values (> 10e7)
 		* or NaN values. If so, it finishes the execution informing the algorithm has diverged. 
 		*/
		void sanityCheck();

		/**
 		* When using Replacing traces, all values not related to the current action are set to 0, while the
 		* values for the current action that their features are active are set to 1. The traces decay following
 		* the rule: e[action][i] = gamma * lambda * e[action][i]. It is possible to also define thresholding.
 		*/
		void decayTrace();

		void updateTrace(int action, float alpha);

		void updateWeights(int action, float alpha, float delta_q);
		/**
 		* Prints the weights in a file. Each line will contain a weight.
 		*/
		void dumpWeights();
		/**
 		* Loads the weights saved in a file. Each line will contain a weight.
 		*/		
		void loadWeights();
	public:
		TrueOnlineSarsaLearner(Environment<FeatureType>& env, Parameters *param);
		/**
		* Implementation of an agent controller. This implementation is True Online Sarsa(lambda).
 		*
 		* @param ALEInterface& ale Arcade Learning Environment interface: object used to define agents'
 		*        actions, obtain simulator's screen, RAM, etc.
 		* @param Features *features object that defines what feature function that will be used.
 		*/
		void learnPolicy(Environment<FeatureType>& env);
		/**
 		* After the policy was learned it is necessary to evaluate its quality. Therefore, a given number
 		* of episodes is run without learning (the vector of weights and the trace are not updated).
 		*
 		* @param ALEInterface& ale Arcade Learning Environment interface: object used to define agents'
 		*        actions, obtain simulator's screen, RAM, etc.
 		*/
		double evaluatePolicy(Environment<FeatureType>& env);
		/**
		* Destructor, not necessary in this class.
		*/
		~TrueOnlineSarsaLearner();
};

#endif
