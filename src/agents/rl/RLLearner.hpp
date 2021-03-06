/****************************************************************************************
** Abstract class that needs to be implemented by any ALE agent. It defines the method 
** that controls the agent, forcing it to act. By using it we ensure that any approach,
** based in RL can be easily run, following the same pattern.
** 
** Author: Marlos C. Machado
***************************************************************************************/

#ifndef RLAGENT_H
#define RLAGENT_H

#include <random>
#include "../Agent.hpp"
#include "../../environments/Environment.hpp"
#include "../../common/Mathematics.hpp"

template<typename FeatureType>
class RLLearner : public Agent<FeatureType>{
	protected:
		ActionVect actions;

		float gamma, epsilon;
		float firstReward;
		float degreeOfOptimism;
		bool   sawFirstReward;

		int    toUseOnlyRewardSign, toBeOptimistic;
		int    randomActionTaken, numActions;
		int    episodeLength, numEpisodesEval;
		int    totalNumberOfFramesToLearn;
        std::mt19937 agentRand;

		/**
 		* It acts in the environment and makes the proper operations in the reward signal (normalizing,
 		* being optimistic, etc).
 		*
 		* @param ALEInterface& ale Arcade Learning Environment interface: object used to define agents'
 		*        actions, obtain simulator screen, RAM, etc.
 		* @param int action action to be taken
 		*
 		* @param vector<double>& reward this vector is used to return, by reference, the reward observed
 		* by executing the action defined as parameter. This vector has two positions: in the first position
 		* the reward to be used by the RL algorithm is returned; in the second position, the game score is
 		* returned.
 		*/
		void act(Environment<FeatureType>& env, int action, std::vector<float> &reward);

		/**
 		* Implementation of an epsilon-greedy function. Epsilon is defined in the constructor,
 		* in the argument Parameters *param
 		*
 		* @return int action to be taken
 		*/
		int epsilonGreedy(std::vector<float> &QValues);

		/**
 		* In RL the Q-values (one per action) are generally updated as the sum of weights for that given action.
 		* To avoid writing it more than once on the code, its update was extracted to a separate function.
 		* It updates the vector<float> Q assuming that vector<int> F is filled, as it sums just the weights
 		* that are active in F.
 		*/
		void updateQValues(std::vector<int> &Features, std::vector<std::vector<float> > &w, std::vector<float> &QValues);

		/**
		* Constructor to be used by the RL classes to save the parameters that
		* will be used by other methods.
		*
		* @param ALEInterface& ale Arcade Learning Environment interface: object used to define agents'
 		*        actions, obtain simulator's screen, RAM, etc.
 		* @param Parameters *param object containing the parameters passed to the algorithm, both by
 		*        file and command line.
 		*
		*/
		RLLearner(Environment<FeatureType>& env, Parameters *param, int seed);

	public:
	   /**
 		* Pure virtual method that needs to be implemented by any RL agent.
 		*
 		* TODO: it may be useful to return something for the caller, as the total reward or policy. 
 		*       Additionally, it may be important to persist what was learned, maybe with another
 		*       pure virtual method?
 		*
 		* @param ALEInterface& ale Arcade Learning Environment interface: object used to define agents'
 		*        actions, obtain simulator's screen, RAM, etc.
 		*/
		virtual void learnPolicy(Environment<FeatureType>& env) = 0;

		/**
 		* Pure virtual method that needs to be implemented by any agent. Once the agent learned a
 		* policy it executes this policy for a given number of episodes. The policy is stored in
 		* the class' object.
 		*
 		* TODO it may be useful to return something for the caller, as an indicator of performance. 
 		*
 		* @param ALEInterface& ale Arcade Learning Environment interface: object used to define agents'
 		*        actions, obtain simulator screen, RAM, etc.
 		*/
		virtual double evaluatePolicy(Environment<FeatureType>& env) = 0;

		/**
		* Destructor, not necessary in this class.
		*/
		virtual ~RLLearner(){};
};

#include "RLLearner.cpp"

#endif
