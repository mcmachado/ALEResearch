/****************************************************************************************
** Starting point for running my algorithm. In this code, the agent first plays some   **
** random trajectories keeping track of the frequency of each feature. Then he defines **
** what are interesting events (feature flips) that should be added to the table that  **
** will have its dimensionality reduced. Once the dimensionality is reduced, each of   **
** the obtained eigen-vectors are used to generate a reward function that will be used **
** to learn options. These options will then be added to the agent's action set and    **
** the process will start over.                                                        **
**                                                                                     **
** Author: Marlos C. Machado                                                           **
*****************************************************************************************/

#include <iterator>     // ostream_operator

#include <ale_interface.hpp>

#include "control/Agent.hpp"
#include "svd/DimReduction.hpp"
#include "common/Parameters.hpp"
#include "control/ControlAgent.hpp"

using namespace std;
using namespace Eigen;

/*TODO: Parametrize it in the agent class. Right now this is being defined
        in this file as well as others, e.g. Agent.cpp*/
#define NUM_FEATURES_NOVELTY 1024

void initializeALE(ALEInterface &ale, const Parameters param){
	ale.setInt  ("random_seed"               , param.seed);
	ale.setInt  ("max_num_frames_per_episode", param.episodeLength);
	ale.setBool ("sound"                     , param.display);
	ale.setBool ("display_screen"            , param.display);
	ale.setFloat("stochasticity"             , 0.00);
	/*In a first moment I am not using this because I need to see the
	  intermediate screens. If I use this I will just see every 5th
	  screen, if a bit flips and unflips between these 5 frames I will
	  not see it. I don't think it is a good idea. I may revisit this
	  assumption later, testing it.*/
	//ale.setFloat("frame_skip"                , param.numStepsPerAction);
	
	ale.loadROM(param.romPath.c_str());
}

void copyDataToEigenStructure(vector<vector<bool> > &vectorData, MatrixXi &eigenData){
	for(int i = 0; i < vectorData[0].size(); i++){
		for(int j = 0; j < vectorData.size(); j++){
			eigenData(i,j) = vectorData[j][i];
		}
	}
	assert(vectorData[0].size() == eigenData.rows());
	assert(vectorData.size() == eigenData.cols());
}

void readCSVFile(string fileName, vector<vector<bool> > &dataset){

	int lineNumber = 0;
	ifstream infile(fileName.c_str());
	while (infile){

		for(int i = 0; i < NUM_FEATURES_NOVELTY * 2; i++){
			dataset[i].push_back(false);
		}

		string s;
		if (!getline( infile, s )) break;

		istringstream ss( s );
		vector <int> record;
		vector<bool> tmp(NUM_FEATURES_NOVELTY * 2, false);

		while (ss){
			string s;
			if (!getline( ss, s, ',' )) break;
			record.push_back( atoi(s.c_str()) );
		}
		for(int i = 0; i < record.size(); i++){
			dataset[record[i]][lineNumber] = true;
		}

		record.clear();
		lineNumber++;
	}
}

int main(int argc, char** argv){

	//What I really need to know:
	Parameters param(argc, argv);
	srand(param.seed);
	ALEInterface ale;
	initializeALE(ale, param);
	Agent agent(ale, &param);

	/*This first vector contains the dataset after obtaining the novelty information. I will then copy it to a
	  Matrix (from Eigen). It is much easier to dynamically allocate vectors, therefore, I decided to do it and
	  then copy the matrix later. To ease the process, this matrix has NUM_FEATURES_NOVELTY * 2 rows and ??
	  columns, to be defined by the number of intereting events. I'll transpose this matrix when copying it to
	  the new data structure.*/
	vector<vector<bool> > dataset(NUM_FEATURES_NOVELTY * 2, vector<bool>());
	/* This vector will contain the information required to center each of the vectors I will receive while
	   learning. I will also use it to center my matrix before running the SVD, and in fact it is filled in there.*/
	vector<float> meansVector;
	vector<float> stdsVector;
	/* This matrix contains the K (defined as a parameter) eigen-vectors that are obtained after running the SVD.
	    Each row contains an eigen-vector, and each eigen-vector will originate an option.*/
	vector<vector<float> > eigenVectors (param.numNewOptionsPerIter, vector<float>(NUM_FEATURES_NOVELTY * 2, 0));

	//readCSVFile("../data/iter1/events/freeway_bits.csv", dataset); //This can be used for testing

	for(int iter = 0; iter < param.maxNumIterations; iter++){
		/* We play randomly using the actions in the action set (primitive actions and options) checking for rare
		  feature flips. The events we observe are called eigen-events, and it is returned in dataset (bit flips).*/
		gatherSamplesFromRandomTrajectories(ale, &param, agent, iter, dataset);

		/* Now I am going to create the Eigen::Matrix and then copy the obtained eigen-events (stored in dataset)
		 to this new structure. It is necessary to run the Singular Value Decomposition. */
		assert(dataset.size() > 0);
		assert(dataset[0].size() > 0);

		MatrixXi datasetOfEigenEvents(dataset[0].size(), dataset.size());
		copyDataToEigenStructure(dataset, datasetOfEigenEvents);
		/* Now we can run SVD on my dataset, to obtain the top K eigen-vectors (this is defined by the eigen-values).
		 Each of these eigen-vectors, assisted by the centering vector, will be responsible to define the reward
		 function that will be used to learn one option. */
		reduceDimensionalityOfEvents(datasetOfEigenEvents, meansVector, stdsVector, 
			eigenVectors, param.numNewOptionsPerIter, iter);

		/* Finally, we can now learn the options using the obtained eigen-vectors and the centering vector. This is
		done param.numNewOptionsPerIter times (it can be done in parallel or sequentially). Memory may be an issue
		if one decides to learn each option in a thread, depending on the size of the feature set. TODO: To allow
		this to be done, 5 ALE's need to be instantiated. Maybe we cannot use the ALE we instantiated above. */
		learnOptionsDerivedFromEigenEvents();

		/* Now we have to clean everything for a second iteration. Everything that is useful for replication should
		have been properly saved in the adequate methods. */
		stdsVector.clear();
		meansVector.clear();
	}
	return 1;
}
