#include <vector>

#include "Parameters.hpp"
#include "../features/BPROFeatures.hpp"

class Learner{
	private:
		//RAMFeatures ramFeatures;
		BPROFeatures bproFeatures;
		std::vector<int> F, Fnext;

		ActionVect actions;

		unsigned int maxFeatVectorNorm;
		int randomActionTaken;
		int numFeatures, currentAction, nextAction;
		int numBasicActions, numTotalActions, numOptions;
		float firstReward;
		bool sawFirstReward;
		float r_alg, r_real;
		float delta, elapsedTime;
		float cumReward, prevCumReward;
		
		struct timeval tvBegin, tvEnd, tvDiff;
		std::string pathToSaveLearnedWeights;

		std::vector<int> optionBeingPlayed;

		std::vector<float> Q, Qnext;
		std::vector<std::vector<float> > e;      //Eligibility trace
		std::vector<std::vector<float> > w;      //Theta, weights vector
		std::vector<std::vector<int> >nonZeroElig;//To optimize the implementation

		void sanityCheck();
		void saveWeightsToFile(std::string suffix);
		int epsilonGreedy(std::vector<float> &QValues);
		void updateReplTrace(int action);
		void updateQValues(std::vector<int> &Features, std::vector<float> &QValues);
		void act(ALEInterface& ale, int action, std::vector<std::vector<std::vector<float> > > &learnedOptions);
		int actionFromOptions(std::vector<int> &Features, 
			std::vector<std::vector<std::vector<float> > > &learnedOptions);
		int getNextAction(std::vector<int> &Features, std::vector<float> &QValues, 
			int episode, std::vector<std::vector<std::vector<float> > > &learnedOptions);

	public:
		Learner(ALEInterface& ale, Parameters *param);
		void learnPolicy(ALEInterface& ale, std::vector<std::vector<std::vector<float> > > &learnedOptions);
		void evaluatePolicy(ALEInterface& ale, std::vector<std::vector<std::vector<float> > > &learnedOptions);
};
