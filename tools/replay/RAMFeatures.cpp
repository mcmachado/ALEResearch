/****************************************************************************************
** Implementation of RAM Features, described in details in the paper below. 
**       "The Arcade Learning Environment: An Evaluation Platform for General Agents.
**        Marc G. Bellemare, Yavar Naddaf, Joel Veness, and Michael Bowling.
**        Journal of Artificial Intelligence Research, 47:253–279, 2013."
**
** The idea is to get the RAM state and each bit be a feature in the feature vector.
**
** Author: Marlos C. Machado
***************************************************************************************/

#include "RAMFeatures.hpp"

#define BITS_IN_BYTE    8
#define BYTES_RAM     128
#define BITS_RAM     1024

typedef unsigned char byte_t;

RAMFeatures::RAMFeatures(){
}

void RAMFeatures::getCompleteFeatureVector(const ALEScreen &screen, const ALERAM &ram, std::vector<bool>& features){	
	assert(features.size() == 0); //If the vector is not empty this can be a mess
	//Get vector with active features:
	std::vector<int> temp;
	std::vector<int>& t = temp;
	this->getActiveFeaturesIndices(screen, ram, t);
	//Iterate over vector with all features storing the non-zero indices in the new vector:
	features = std::vector<bool>(this->getNumberOfFeatures(), 0);
	for(unsigned int i = 0; i < t.size(); i++){
		features[t[i]] = 1;
	}
}

void RAMFeatures::getActiveFeaturesIndices(
	const ALEScreen &screen, const ALERAM &ram, std::vector<int>& features){
	assert(features.size() == 0); //If the vector is not empty this can be a mess
	byte_t byte;
	char output[BITS_IN_BYTE];

	int pos = 0;
	for(int i = 0; i < BYTES_RAM; i++){
		//Decomposing byte in bits
		byte = ram.get(i);		
    	for (int b = 0; b < BITS_IN_BYTE; b++) {
  			output[b] = (byte >> b) & 1;
		}
    	//Saving bits in feature vector (little endian)
    	for(int b = 0; b < BITS_IN_BYTE; b++){
    		if(output[b]){
    			features.push_back(pos++);
    		}
    		else{
    			pos++;
    		}
    	}
	}
	//Bias:
	features.push_back(BITS_RAM);
}

int RAMFeatures::getNumberOfFeatures(){
	return BITS_RAM + 1;
}

RAMFeatures::~RAMFeatures(){}
