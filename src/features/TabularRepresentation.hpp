/**
 * @file   RiverSwimFeatures.hpp
 * @author Marlos C. Machado
 * 
 * @brief  This file provides feature classes for the river swim environment
 * 
 */

#ifndef TABULAR_H
#define TABULAR_H

#include <vector>
#include "../environments/simple_mdps/4_state.hpp"

class TabularRepresentation;

class TabularRepresentation{
public:
    typedef bool FeatureType;
    
    TabularRepresentation(){}

    void getCompleteFeatureVector(FourStatesEnvironment<TabularRepresentation>* env, std::vector<bool>& features){
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

    int getNumberOfFeatures(FourStatesEnvironment<TabularRepresentation>* env){
        return env->getNumberOfFeatures();
    }

    void getActiveFeaturesIndices(FourStatesEnvironment<TabularRepresentation>* env, std::vector<int>& features){
        assert(features.size() == 0); //If the vector is not empty this can be a mess
        features.push_back(env->getCurrentState());
    }
};

#endif
