/**
 * @file   RiverSwimFeatures.hpp
 * @author Marlos C. Machado
 * 
 * @brief  This file provides feature classes for the river swim environment
 * 
 */

#ifndef RIVERSWIMFEAT_H
#define RIVERSWIMFEAT_H

#include <vector>
#include "../environments/river_swim/RiverSwimEnvironment.hpp"

class RiverSwimFeatures;

class RiverSwimFeatures{
public:
    typedef bool FeatureType;
    
    RiverSwimFeatures(){}

    void getCompleteFeatureVector(RiverSwimEnvironment<RiverSwimFeatures>* env, std::vector<bool>& features){
        assert(features.size() == 0); //If the vector is not empty this can be a mess
        //Get vector with active features:
        std::vector<int> temp;
        std::vector<int>& t = temp;
        this->getActiveFeaturesIndices(env, t);
        //Iterate over vector with all features storing the non-zero indices in the new vector:
        features = std::vector<bool>(env->getNumberOfFeatures(), 0);
        for(unsigned int i = 0; i < t.size(); i++){
            features[t[i]] = 1;
        }
    }

    int getNumberOfFeatures(RiverSwimEnvironment<RiverSwimFeatures>* env){
        return env->getNumberOfFeatures();
    }

    void getActiveFeaturesIndices(RiverSwimEnvironment<RiverSwimFeatures>* env, std::vector<int>& features){
        assert(features.size() == 0); //If the vector is not empty this can be a mess
        features.push_back(env->getCurrentState());
        /*features.push_back(env->getNumberOfFeatures() - 1);*/ //This is a tabular representation, no bias term is necessary
    }
};

#endif
