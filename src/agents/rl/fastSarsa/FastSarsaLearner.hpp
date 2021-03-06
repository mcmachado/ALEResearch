/****************************************************************************************
** Author: Marlos C. Machado
***************************************************************************************/

#ifndef SARSALEARNER_H
#define SARSALEARNER_H

#include "../RLLearner.hpp"
#include <vector>
#include <fstream>

class FastSarsaLearner : public RLLearner<bool>{
	private:

		float alpha, delta, lambda, traceThreshold;
		int numFeatures, currentAction, nextAction;
		int kBound, toSaveCheckPoint, saveWeightsEveryXSteps;
		int toSaveWeightsAfterLearning;

		std::string nameWeightsFile, pathWeightsFileToLoad;
		std::string checkPointName;
        std::string nameForLearningCondition;
        int episodePassed, numEpisodesEval, numEpisodesLearn;
        int totalNumberFrames;
        unsigned int maxFeatVectorNorm;

		std::vector<int> F;						  //Set of features active
		std::vector<int> Fnext;              	  //Set of features active in next state
		std::vector<float> Q;               	  //Q(a) entries
		std::vector<float> Qnext;           	  //Q(a) entries for next action
		std::vector<std::vector<float> > w;       //Theta, weights vector
		std::vector<std::vector<int> >nonZeroElig;//To optimize the implementation
		std::vector<std::vector<int> > featureSeen;

		/**
 		* Constructor declared as private to force the user to instantiate SarsaLearner
 		* informing the parameters to learning/execution.
 		*/
		FastSarsaLearner();
		/**
 		* This method evaluates whether the Q-values are sound. By unsound I mean huge Q-values (> 10e7)
 		* or NaN values. If so, it finishes the execution informing the algorithm has diverged. 
 		*/
		void sanityCheck();
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
		FastSarsaLearner(Environment<bool>& env, Parameters *param, int seed);
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
		~FastSarsaLearner();
};

#endif
