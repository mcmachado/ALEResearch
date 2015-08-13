/*******************************************************************************
*** This implements an algorithm that plays randomly keeping track of the     **
*** frequency of the features it has seen. When a feature transition becomes  **
*** rare (defined via threshold parameter), and the agent sees it, it prints  **
*** the indicator function for that coordinate. It can load weights           **
*** (representing options) as well. If weights are loaded they are treated    **
*** as primitive actions that will be used by the random agent as well.       **
*******************************************************************************/

#include <vector>

#include <ale_interface.hpp>

#include "control.hpp"
#include "constants.hpp"
#include "Parameters.hpp"
#include "../features/BPROFeatures.hpp"
#include "../features/RAMFeatures.hpp"

vector<double> frequency;         //[0:1023] transitions 0->1; [1024:2048] transitions 1->0

void updateAverage(vector<bool> Fprev, vector<bool> F, int frame, Parameters param, int gameId){
	assert (Fprev.size() == F.size());
	assert (F.size() == NUM_BITS);
	
	bool toPrint = false;

	vector<int> tempVector(2 * NUM_BITS, 0);

	for(int i = 0; i < NUM_BITS; i++){
		if(!Fprev[i] && F[i]){ // 0->1
			frequency[i] = (frequency[i] * (frame - 1) + 1) / frame;
			tempVector[i] = 1;
			if(frame > FRAMES_TO_WAIT && frequency[i] < param.freqThreshold){
				tempVector[i] = 2; //1 denotes flip, 2 denotes relevant flip
				toPrint = true;
			}
		} else{
			frequency[i] = (frequency[i] * (frame - 1) + 0) / frame;
		}		
		if(Fprev[i] && !F[i]){ // 1->0
			frequency[i + NUM_BITS] = (frequency[i + NUM_BITS] * (frame - 1) + 1) / frame;
			tempVector[i + NUM_BITS] = 1;
			if(frame > FRAMES_TO_WAIT && frequency[i + NUM_BITS] < param.freqThreshold){
				tempVector[i + NUM_BITS] = 2;
				toPrint = true;
			}
		} else{
			frequency[i + NUM_BITS] = (frequency[i + NUM_BITS] * (frame - 1) + 0) / frame;
		}
	}

	ofstream myFileBits, myFileBytes;
	myFileBits.open (param.outputPath + "_bits.csv", ios::app);
	if(toPrint){
		for(int i = 0; i < tempVector.size(); i++){
			if(param.toReportAll == 1 && tempVector[i] != 0){
				myFileBits << i << ",";
			}
			if(param.toReportAll == 0 && tempVector[i] == 2){
				myFileBits << i << ",";
			}
		}
		myFileBits << endl;
	}
	myFileBits.close();
}

int actUpdatingAvg(ALEInterface& ale, RAMFeatures *ram, BPROFeatures *features, int nextAction, 
	vector<vector<vector<float> > > &w, Parameters param, int totalNumFrames, int gameId,
	vector<bool> &F, vector<bool> &Fprev){

	int reward = 0;

	//If the selected action was one of the primitive actions
	if(nextAction < NUM_ACTIONS){ 
		for(int i = 0; i < FRAME_SKIP && totalNumFrames + ale.getEpisodeFrameNumber() < MAX_NUM_FRAMES; i++){
			reward += ale.act((Action) nextAction);
			Fprev.swap(F);
			F.clear();
			ram->getCompleteFeatureVector(ale.getRAM(), F);
			F.pop_back();
			updateAverage(Fprev, F, ale.getEpisodeFrameNumber(), param, gameId);
		}
	}
	//If the selected action was one of the options
	else{
		int currentAction;
		vector<int> Fbpro;	                  //Set of features active
		vector<float> Q(NUM_ACTIONS, 0.0);    //Q(a) entries

		int option = nextAction - NUM_ACTIONS;
		while(rand()%1000 > 1000 * PROB_TERMINATION && !ale.game_over() && totalNumFrames + ale.getEpisodeFrameNumber() < MAX_NUM_FRAMES){
			//Get state and features active on that state:		
			Fbpro.clear();
			features->getActiveFeaturesIndices(ale.getScreen(), Fbpro);
			updateQValues(Fbpro, Q, w, option);       //Update Q-values for each possible action
			currentAction = epsilonGreedy(Q);
			//Take action, observe reward and next state:
			reward += ale.act((Action) currentAction);
			Fprev.swap(F);
			F.clear();
			ram->getCompleteFeatureVector(ale.getRAM(), F);
			F.pop_back();
			updateAverage(Fprev, F, ale.getEpisodeFrameNumber(), param, gameId);
		}
	}
	return reward;
}

int playGame(ALEInterface& ale, RAMFeatures *ram, BPROFeatures *bpro, 
	vector<vector<vector<float> > > &w, Parameters param, int totalNumFrames, int gameId){
	ale.reset_game();
	vector<bool> F(NUM_BITS, 0); //Set of active features
	vector<bool> Fprev;

	int score = 0;
	while(!ale.game_over() && totalNumFrames + ale.getEpisodeFrameNumber() < MAX_NUM_FRAMES){
		int nextAction = getNextAction(ale, param.numOptions);
		score += actUpdatingAvg(ale, ram, bpro, nextAction, w, param, totalNumFrames, gameId, F, Fprev);
	}
	totalNumFrames += ale.getEpisodeFrameNumber();
	printf("Episode: %d, Final score: %d, Total Num. Frames: %d\n", gameId+1, score, totalNumFrames);
	return totalNumFrames;
}

void loadWeights(Parameters param, BPROFeatures *features, vector<vector<vector<float> > > &w){
	
	int numFeatures = features->getNumberOfFeatures();

	for(int i = 0; i < param.numOptions; i++){
		w.push_back(vector< vector<float> >(NUM_ACTIONS, vector<float>(numFeatures, 0.0)));
	}

	for(int i = 0; i < param.optionsWgts.size(); i++){
		string line;
		int nActions, nFeatures;
		int j, k;
		float value;

		std::ifstream weightsFile (param.optionsWgts[i].c_str());

		weightsFile >> nActions >> nFeatures;
		assert(nActions == NUM_ACTIONS);
		assert(nFeatures == numFeatures);

		while(weightsFile >> j >> k >> value){
			w[i][j][k] = value;
		}
	}
}

int main(int argc, char** argv){
	Parameters param(argc, argv);
	srand(param.seed);

	RAMFeatures ramFeatures;
	BPROFeatures bproFeatures(param.gameName);

	vector<vector<vector<float> > > w;

	loadWeights(param, &bproFeatures, w);

	ALEInterface ale(0);
	ale.setInt  ("random_seed"               , param.seed);
	ale.setInt  ("max_num_frames_per_episode", 18000     );
	ale.setBool ("color_averaging"           , true      );
	ale.setFloat("frame_skip"                , 1         );
	ale.setFloat("repeat_action_probability" , 0.00      );

	ale.loadROM(param.romPath.c_str());

	for(int i = 0; i < 2 * NUM_BITS; i++){
		frequency.push_back(0.0);
	}

	int totalNumFrames = 0;
	for(int episode = 0; totalNumFrames < MAX_NUM_FRAMES; episode++){
		totalNumFrames = playGame(ale, &ramFeatures, &bproFeatures, w, param, totalNumFrames, episode);
		totalNumFrames += ale.getEpisodeFrameNumber();
	}

	return 0;
}
