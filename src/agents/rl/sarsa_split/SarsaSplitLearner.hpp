/****************************************************************************************
** Implementation of Sarsa(lambda). It implements Fig. 8.8 (Linear, gradient-descent 
** Sarsa(lambda)) from the book "R. Sutton and A. Barto; Reinforcement Learning: An 
** Introduction. 1st edition. 1988."
** Some updates are made to make it more efficient, as not iterating over all features.
** 
** Author: Marlos C. Machado
***************************************************************************************/

#ifndef SARSASPLITLEARNER_H
#define SARSASPLITLEARNER_H

#include "../RLLearner.hpp"
#include <vector>
#include <fstream>

class SarsaSplitLearner : public RLLearner<bool>{
	private:

		float alphaW, deltaW, deltaPsi, lambdaW, traceThreshold;
		float alphaPsi, lambdaPsi, gammaPsi;
		int numFeatures, currentAction, nextAction;
		int toSaveWeightsAfterLearning, saveWeightsEveryXSteps, toSaveCheckPoint;

		std::string nameWeightsFile, pathWeightsFileToLoad;
		std::string checkPointName;
        std::string nameForLearningCondition;
        int episodePassed, numEpisodesEval, numEpisodesLearn;
        int totalNumberFrames, episodeLength;
        unsigned int maxFeatVectorNorm;

		std::vector<int> F;				  	       //Set of features active
		std::vector<int> Fnext;			           //Set of features active in next state
		std::vector<std::vector<int> > Fcount;     //Used to calculate the step sizes
		std::vector<float> QW, QPsi, Q;            //Q(a) entries
		std::vector<float> QnextW, QnextPsi;       //Q(a) entries for next action
		std::vector<std::vector<float> > eW, ePsi; //Eligibility trace
		std::vector<std::vector<float> > w;        //Theta, weights vector
		std::vector<std::vector<float> > psi;      //Psi, auxiliary Q-values
		std::vector<std::vector<int> >nonZeroEligW;//To optimize the implementation
		std::vector<std::vector<int> >nonZeroEligPsi;
		std::vector<std::vector<int> > featureSeen;

		/**
 		* Constructor declared as private to force the user to instantiate SarsaSplitLearner
 		* informing the parameters to learning/execution.
 		*/
		SarsaSplitLearner();
		/**
 		* This method evaluates whether the Q-values are sound. By unsound I mean huge Q-values (> 10e7)
 		* or NaN values. If so, it finishes the execution informing the algorithm has diverged. 
 		*/
		void sanityCheck(std::vector<float> &QValues);
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
		/**
 		* Prints the weights in a file. Each line will contain a weight.
 		*/
		void saveWeightsToFile(std::string suffix="");
		/**
 		* Loads the weights saved in a file. Each line will contain a weight.
 		*/		
		void loadWeights();
		void saveCheckPoint(int episode, int totalNumberFrames,  std::vector<float>& episodeResults, int& frequency, std::vector<int>& episodeFrames, std::vector<double>& episodeFps);
        void loadCheckPoint(std::ifstream& checkPointToLoad);
	public:
		SarsaSplitLearner(Environment<bool>& env, Parameters *param, int seed);
		/**
 		* Implementation of an agent controller. This implementation is Sarsa(lambda).
 		*
 		* @param ALEInterface& ale Arcade Learning Environment interface: object used to define agents'
 		*        actions, obtain simulator's screen, RAM, etc.
 		* @param Features *features object that defines what feature function that will be used.
 		*/
		void learnPolicy(Environment<bool>& env);
		/**
 		* After the policy was learned it is necessary to evaluate its quality. Therefore, a given number
 		* of episodes is run without learning (the vector of weights and the trace are not updated).
 		*
 		* @param ALEInterface& ale Arcade Learning Environment interface: object used to define agents'
 		*        actions, obtain simulator's screen, RAM, etc.
 		*/
		double evaluatePolicy(Environment<bool>& env);
		/**
		* Destructor, not necessary in this class.
		*/
		~SarsaSplitLearner();
};

#endif
