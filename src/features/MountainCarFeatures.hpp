/**
 * @file   MountainCarFeatures.hpp
 * @author Marlos C. Machado, Nicolas Carion
 * 
 * @brief  This file provides feature classes for the Mountain Car Environment.
 * The tiling code is borrowed from Rich Sutton.
 */

#ifndef MCFEAT_H
#define MCFEAT_H

#include "../environments/mountain_car/MountainCarEnvironment.hpp"
#include <vector>

#define NUM_TILES_X 10
#define NUM_TILES_V 10
#define NUM_TILINGS 10
#define NUM_ACTIVE_FEAT 10

class MountainCarFeatures;

class MountainCarFeatures{

public:
    typedef bool FeatureType;
    
    MountainCarFeatures(){}

    void getCompleteFeatureVector(MountainCarEnvironment<MountainCarFeatures>* env, std::vector<bool>& features){
		assert(features.size() == 0); //If the vector is not empty this can be a mess
		//Get vector with active features:
		std::vector<int> temp;
		std::vector<int>& t = temp;
		this->getActiveFeaturesIndices(env, t);
		//Iterate over vector with all features storing the non-zero indices in the new vector:
		features = std::vector<bool>(this->getNumberOfFeatures(env), 0);
		for(unsigned int i = 0; i < t.size(); i++){
			features[t[i]] = 1;
		}
    }

    int getNumberOfFeatures(MountainCarEnvironment<MountainCarFeatures>*){
        return NUM_TILINGS * NUM_TILES_X * NUM_TILES_V;
    }

    void getActiveFeaturesIndices(MountainCarEnvironment<MountainCarFeatures>* env, std::vector<int>& features){

    	assert(features.size() == 0);

    	float x_max =  0.5;
    	float x_min = -1.2;
    	float v_max =  0.07;
    	float v_min = -0.07;

	    double x_size = (x_max - x_min)/double(NUM_TILES_X - 1);
	    double v_size = (v_max - v_min)/double(NUM_TILES_V - 1);
	    
	    for(int i = 0; i < NUM_ACTIVE_FEAT; i++){
	        double x = env->getPos();
	        double v = env->getVel();
	        
	        int fx = int(floor((x - x_min)/x_size));
	        fx = fmin(fx, NUM_TILES_X); // catch border case
	        int fv = int(floor((v - v_min)/v_size));
	        fv = fmin(fv, NUM_TILES_V); // catch border case

	        int ft = fx + NUM_TILES_X * fv + i * (NUM_TILES_X * NUM_TILES_V);
	        assert(ft >= 0);
	        assert(ft < (NUM_TILINGS * NUM_TILES_X * NUM_TILES_V));
	        
	        features.push_back(ft);
	    }
    }
};

#endif