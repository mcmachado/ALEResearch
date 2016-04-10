/* Author: Marlos C. Machado */

#include "ControlAgent.hpp"
#include "../common/Timer.hpp"

#define NUM_BITS 1024

int playGame(ALEInterface& ale, Parameters *param, Agent &agent, int iter, vector<vector<char> > &dataset){

	int score = 0;
	int frame = 0;
	int totalNumActions = agent.getNumAvailActions();

	ale.reset_game();
	while(!ale.game_over()){
		int nextAction = rand() % totalNumActions;
		score += agent.actStoringObservation(ale, param, frame, nextAction, iter, dataset);
	}

	return score;
}

void gatherSamplesFromRandomTrajectories(ALEInterface& ale, Parameters *param, Agent &agent,
	vector<vector<char> > &dataset, int iter){

	cout << "Generating Samples to Identify Rare Events\n";
	for(int i = 1; i < param->numGamesToSampleRareEvents + 1; i++){
		int finalScore = playGame(ale, param, agent, iter, dataset);
		cout << i << ": Final score: " << finalScore << endl;
	}
}

void learnEigenBehavioursDerivedFromEigenPurposes(ALEInterface &ale, Parameters *param,
	Agent &agent, std::vector<float> &datasetMeans, std::vector<float> &datasetStds,
	std::vector<std::vector<float> > &eigenVectors, int iter){

	cout << "Learning Eigenbehaviour from Eigenpurpose\n";

	/* We are going to learn each eigenbehaviour iteratively. We could do this in parallel,
	   but for sake of simplicity this is being done sequentially now. I may have
	   this to my TODO list, to parallelize this step of the computation. */
	  for(int i = 0; i <  param->numNewEigenBehavioursPerIter; i++){
		cout << "Learning Eigenbehaviour #" << i+1 << endl;
		int currEigenBehaviourIdx = iter * param->numNewEigenBehavioursPerIter + i;
		agent.learnedEigenBehaviours.push_back(vector< vector<float> >(agent.getNumAvailActions(), 
			vector<float>(agent.bproFeatures.getNumberOfFeatures(), 0.0)));

		learnEigenBehaviours(ale, param, agent, datasetMeans, datasetStds, eigenVectors[i], currEigenBehaviourIdx);
	  }
	  agent.numberOfEigenBehaviours += param->numNewEigenBehavioursPerIter;

	/* For checkpointing reasons (and replayability) I am also going to save every
	   set of weights I learned, representing each of the eigenbehaviours. */
	  //TODO: Implement saving weights, this should generate huge files
}

/*TODO: I have too many declarations here. Should I create a class Sarsa?
        Should I add this learning algorithm to my agent? Probably, because
        some of the things are already implemented there. I would also save
        code when I am going to learn the policy that tries to maximize the
        total score. */
void learnEigenBehaviours(ALEInterface &ale, Parameters *param, Agent &agent,
	std::vector<float> &datasetMeans, std::vector<float> &datasetStds,
	std::vector<float> &eigenVectors, int eigenbehaviour){

	vector<vector<float> > e;        //Eligibility trace
	vector<vector<int> >nonZeroElig; //To optimize the implementation

	for(int i = 0; i < agent.getNumAvailActions(); i++){
		e.push_back(vector<float>(agent.bproFeatures.getNumberOfFeatures(), 0.0));
		nonZeroElig.push_back(vector<int>());
	}

	//Performance monitoring:
	double elapsedTime;
	struct timeval tvBegin, tvEnd, tvDiff;
	//Rewards monitoring:
	vector<float> reward;
	bool sawFirstReward = 0;
	float cumIntrReward = 0, prevCumIntrReward = 0, delta = 0;
	float firstReward = 1.0, cumReward = 0, prevCumReward = 0;
	//Variables necessary for acting:
	int numActions = agent.getNumAvailActions();
	vector<float> Q(numActions, 0.0); 		//Q(a) entries
	vector<float> Qnext(numActions, 0.0);   //Q(a) entries for next action
	int currentAction, nextAction;
	
	//Features monitoring:
	vector<int> Fbpro;				//Set of features active
	vector<int> FbproNext;          //Set of features active in next state
	unsigned int maxFeatVectorNorm = 1;
	//Repeat (for each episode):
	int episode, totalNumberFrames = 0;
	int totalNumberOfFramesToLearn = param->learningLength;
	for(episode = 0; totalNumberFrames < totalNumberOfFramesToLearn; episode++){
		agent.cleanTraces(e, nonZeroElig);
		//Obtain new observation:
		Fbpro.clear();
		agent.bproFeatures.getActiveFeaturesIndices(ale.getScreen(), Fbpro);
		agent.updateQValues(agent.learnedEigenBehaviours[eigenbehaviour], Fbpro, Q, eigenbehaviour);
		currentAction = agent.epsilonGreedy(Q, param->epsilon);
		//Repeat(for each step of episode) until game is over (or the maximum number of steps is reached):
		gettimeofday(&tvBegin, NULL);
		while(!ale.game_over()){
			reward.clear();
			reward.push_back(0.0);
			reward.push_back(0.0);
			agent.updateQValues(agent.learnedEigenBehaviours[eigenbehaviour], Fbpro, Q, eigenbehaviour);
			agent.sanityCheck(Q);
			//Take action, observe reward and next state:
			agent.act(ale, currentAction, param, datasetMeans, datasetStds, eigenVectors, reward);
			cumIntrReward += reward[0];
			cumReward  += reward[1];
			if(!ale.game_over()){
				//Obtain active features in the new state:
				FbproNext.clear();
				agent.bproFeatures.getActiveFeaturesIndices(ale.getScreen(), FbproNext);
				agent.updateQValues(agent.learnedEigenBehaviours[eigenbehaviour], FbproNext, Qnext, eigenbehaviour);     //Update Q-values for the new active features
				nextAction = agent.epsilonGreedy(Qnext, param->epsilon);
			}
			else{
				nextAction = 0;
				for(unsigned int i = 0; i < Qnext.size(); i++){
					Qnext[i] = 0;
				}
			}
			//To ensure the learning rate will never increase along
			//the time, Marc used such approach in his JAIR paper
			if (Fbpro.size() > maxFeatVectorNorm){
				maxFeatVectorNorm = Fbpro.size();
			}
			delta = reward[0] + param->gamma * Qnext[nextAction] - Q[currentAction];
			agent.updateReplTrace(param, currentAction, Fbpro, e, nonZeroElig);
			//Update weights vector:
			for(unsigned int a = 0; a < nonZeroElig.size(); a++){
				for(unsigned int i = 0; i < nonZeroElig[a].size(); i++){
					int idx = nonZeroElig[a][i];
					agent.learnedEigenBehaviours[eigenbehaviour][a][idx] += (param->alpha/maxFeatVectorNorm) * delta * e[a][idx];
				}
			}
			Fbpro = FbproNext;
			currentAction = nextAction;
		}
		gettimeofday(&tvEnd, NULL);
		timeval_subtract(&tvDiff, &tvEnd, &tvBegin);
		elapsedTime = double(tvDiff.tv_sec) + double(tvDiff.tv_usec)/1000000.0;

		double fps = double(ale.getEpisodeFrameNumber())/elapsedTime;
		printf("episode: %d,\t%.0f points,\tavg. return: %.1f,\tnovelty reward: %.2f (%.2f),\t%d frames,\t%.0f fps\n",
			episode + 1, cumReward - prevCumReward, (double)cumReward/(episode + 1.0),
			cumIntrReward - prevCumIntrReward, cumIntrReward/(episode + 1.0), ale.getEpisodeFrameNumber(), fps);
		totalNumberFrames += ale.getEpisodeFrameNumber();
		prevCumReward = cumReward;
		prevCumIntrReward = cumIntrReward;
		ale.reset_game();
	}
}
