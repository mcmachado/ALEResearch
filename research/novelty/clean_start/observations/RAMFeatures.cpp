/* Author: Marlos C. Machado */

#include "RAMFeatures.hpp"

using namespace std;

#define BITS_IN_BYTE    8
#define BYTES_RAM     128
#define BITS_RAM     1024

typedef unsigned char byte_t;

RAMFeatures::RAMFeatures(){}

RAMFeatures::~RAMFeatures(){}

void RAMFeatures::getCompleteFeatureVector(const ALERAM &ram, vector<bool>& features){	
	assert(features.size() == 0); //If the vector is not empty this can be a mess
	//Get vector with active features:
	vector<int> temp;
	vector<int>& t = temp;
	this->getActiveFeaturesIndices(ram, t);
	//Iterate over vector with all features storing the non-zero indices in the new vector:
	features = vector<bool>(this->getNumberOfFeatures(), 0);
	for(unsigned int i = 0; i < t.size(); i++){
		features[t[i]] = 1;
	}
}

void RAMFeatures::getActiveFeaturesIndices(const ALERAM &ram, vector<int>& features){
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
