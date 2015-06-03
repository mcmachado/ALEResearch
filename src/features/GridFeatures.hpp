/**
 * @file   GridFeatures.hpp
 * @author Nicolas Carion
 * @date   Wed Jun  3 08:56:48 2015
 * 
 * @brief  This file provides feature classes for the grid environments
 * 
 * 
 */
#ifndef GRIDFEAT_H
#define GRIDFEAT_H

#include "../environments/grid/GridEnvironment.hpp"
#include <vector>
class BasicGridFeatures;
class BasicGridFeatures
{
public:
    typedef bool FeatureType;
    
    BasicGridFeatures(){}

    void getCompleteFeatureVector(GridEnvironment<BasicGridFeatures>* env,std::vector<bool>& features){
        auto size = env->getGridSize();
        features.clear();
        for(int i=0;i<size.first;i++){
            for(int j=0;j<size.first;j++){
                features.push_back(true);
            }
        }
    }

    int getNumberOfFeatures(GridEnvironment<BasicGridFeatures>* env){
        auto size = env->getGridSize();
        return (1+size.first)*(size.second+1);
    }

    void getActiveFeaturesIndices(GridEnvironment<BasicGridFeatures>* env,std::vector<std::pair<int,bool>>& active){
        auto size = env->getGridSize();
        auto pos = env->getPos();
        active.clear();
        active.push_back({pos.second*size.first+pos.first,true});
    }
};



#endif
