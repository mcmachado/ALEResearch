#include <vector>

#include "Parameters.hpp"
#include "../features/RAMFeatures.hpp"
#include "../features/BPROFeatures.hpp"

class Learner{
	private:
		RAMFeatures ramFeatures;
		BPROFeatures bproFeatures;
		std::vector<int> F, Fnext;
		std::vector<bool> FRam, FnextRam;
		std::vector<float> transitions;

		ActionVect actions;

		unsigned int maxFeatVectorNorm;
		int numFeatures, currentAction, nextAction;
		int numBasicActions, numTotalActions, numOptions;

		float delta, elapsedTime;
		float cumReward, prevCumReward;
		float cumIntrReward, prevCumIntrReward;
		
		struct timeval tvBegin, tvEnd, tvDiff;

		std::string nameWeightsFile, pathWeightsFileToLoad;
		std::string pathToRewardDescription, pathToStatsDescription;

		std::vector<float> std;
		std::vector<float> mean;
		std::vector<float> eigVector;

		std::vector<float> Q, Qnext;
		std::vector<std::vector<float> > e;      //Eligibility trace
		std::vector<std::vector<float> > w;      //Theta, weights vector
		std::vector<std::vector<int> >nonZeroElig;//To optimize the implementation

		void sanityCheck();
		void saveWeightsToFile(std::string suffix);
		int epsilonGreedy(std::vector<float> &QValues);
		void updateReplTrace(int action, std::vector<int> &Features);
		void updateTransitionVector(vector<bool> F, vector<bool> Fnext);
		void updateQValues(std::vector<int> &Features, std::vector<float> &QValues);
		void act(ALEInterface& ale, int action, std::vector<float> &reward, 
			std::vector<std::vector<std::vector<float> > > &learnedOptions);
		int playOption(ALEInterface& ale, int option, 
			std::vector<std::vector<std::vector<float> > > &learnedOptions);
	public:
		Learner(ALEInterface& ale, Parameters *param);
		void learnPolicy(ALEInterface& ale, std::vector<std::vector<std::vector<float> > > &learnedOptions);
};