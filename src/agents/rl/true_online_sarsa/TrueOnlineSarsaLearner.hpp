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

#ifndef TOSARSALEARNER_H
#define TOSARSALEARNER_H
#include "../RLLearner.hpp"
#include <vector>

template<typename FeatureType>
class TrueOnlineSarsaLearner : public RLLearner<FeatureType>{
	private:
		double alpha, delta, lambda, traceThreshold;
		int numFeatures, currentAction, nextAction;

		std::string nameWeightsFile;

        vector<pair<int,FeatureType> > F;					//Set of features active (along with their values)
        vector<pair<int,FeatureType> > Fnext;              //Set of features active in next state
		vector<double> Q;               //Q(a) entries
		vector<double> Qnext;           //Q(a) entries for next action
		vector<vector<double> > e;      //Eligibility trace
		vector<vector<double> > w;      //Theta, weights vector
		vector<vector<int> >nonZeroElig;//To optimize the implementation   

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
 		* In Sarsa the Q-values (one per action) are updated as the sum of weights for that given action.
 		* To avoid writing it more than once on the code, its update was extracted to a separate function.
 		* It updates the vector<double> Q assuming that vector<int> F is filled, as it sums just the weights
 		* that are active in F.
 		*/
    void updateQValues(vector<pair<int,FeatureType> > &Features, vector<double> &QValues);
		/**
 		* When using Replacing traces, all values not related to the current action are set to 0, while the
 		* values for the current action that their features are active are set to 1. The traces decay following
 		* the rule: e[action][i] = gamma * lambda * e[action][i]. It is possible to also define thresholding.
 		*/
		void decayTrace();

		void updateTrace(int action, double alpha);

		void updateWeights(int action, double alpha, double delta_q);
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
 		* Implementation of an agent controller. This implementation is Sarsa(lambda).
 		*
 		* TODO it may be useful to return something for the caller, as the total reward or policy. 
 		*
 		* @param Environment<FeatureType>& env Arcade Learning Environment interface: object used to define agents'
 		*        actions, obtain simulator's screen, RAM, etc.
 		*/
    void learnPolicy(Environment<FeatureType>& env);
		/**
 		* After the policy was learned it is necessary to evaluate its quality. Therefore, a given number
 		* of episodes is run without learning (the vector of weights and the trace are not updated).
 		*
 		* @param Environment<FeatureType>& env Arcade Learning Environment interface: object used to define agents'
 		*        actions, obtain simulator's screen, RAM, etc.
 		*/
		void evaluatePolicy(Environment<FeatureType>& env);
		/**
		* Destructor, not necessary in this class.
		*/
		~TrueOnlineSarsaLearner();
};

#include "TrueOnlineSarsaLearner.cpp"

#endif
