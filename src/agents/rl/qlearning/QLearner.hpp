/****************************************************************************************
** Implementation of Q(lambda). It implements Fig. 8.9 (Linear, gradient-descent 
** Q(lambda)) from the book "R. Sutton and A. Barto; Reinforcement Learning: An 
** Introduction. 1st edition. 1988."
** Some updates are made to make it more efficient, as not iterating over all features.
**
** TODO: Make it as efficient as possible. 
** 
** Author: Marlos C. Machado
***************************************************************************************/

#ifndef QLEARNER_H
#define QLEARNER_H
#include "../RLLearner.hpp"
#include <vector>

class QLearner : public RLLearner<bool>{
	private:
		double alpha, delta, lambda, traceThreshold;
		int numFeatures, currentAction, nextAction;

		std::string nameWeightsFile;

		vector<int> F;					//Set of features active
		vector<int> Fnext;              //Set of features active in next state
		vector<double> Q;               //Q(a) entries
		vector<double> Qnext;           //Q(a) entries for next action
		vector<vector<double> > e;      //Eligibility trace
		vector<vector<double> > w;      //Theta, weights vector
		vector<vector<int> >nonZeroElig;//To optimize the implementation
		
		/**
 		* Constructor declared as private to force the user to instantiate QLearner
 		* informing the parameters to learning/execution.
 		*/
		QLearner();

		/**
 		* This method evaluates whether the Q-values are sound. By unsound I mean huge Q-values (> 10e7)
 		* or NaN values. If so, it finishes the execution informing the algorithm has diverged. 
 		*/
		void sanityCheck();

		/**
 		* In Q-Learning the Q-values (one per action) are updated as the sum of weights for that given action.
 		* To avoid writing it more than once on the code, its update was extracted to a separate function.
 		* It updates the vector<double> Q assuming that vector<int> F is filled, as it sums just the weights
 		* that are active in F.
 		*/
		void updateQValues(vector<int> &Features, vector<double> &QValues);
		
		/**
 		* When using Replacing traces, all values not related to the current action are set to 0, while the
 		* values for the current action that their features are active are set to 1. The traces decay following
 		* the rule: e[action][i] = gamma * lambda * e[action][i]. It is possible to also define thresholding.
 		*/
		void updateReplTrace(int action);
	public:
    QLearner(Environment<bool>& env, Parameters *param);
		/**
 		* Implementation of an agent controller. This implementation is Q(lambda).
 		*
 		* TODO it may be useful to return something for the caller, as the total reward or policy. 
 		*
 		* @param Environment& env Arcade Learning Environment interface: object used to define agents'
 		*        actions, obtain simulator's screen, RAM, etc.
 		* @param Features *features object that defines what feature function that will be used.
 		*/
		void learnPolicy(Environment<bool>& env);
		/**
 		* After the policy was learned it is necessary to evaluate its quality. Therefore, a given number
 		* of episodes is run without learning (the vector of weights and the trace are not updated).
 		*
 		* @param Environment& env Arcade Learning Environment interface: object used to define agents'
 		*        actions, obtain simulator's screen, RAM, etc.
 		* @param Features *features object that defines what feature function that will be used.
 		*/
		void evaluatePolicy(Environment<bool>& env);
		/**
		* Destructor, not necessary in this class.
		*/
		~QLearner();
};


#endif
