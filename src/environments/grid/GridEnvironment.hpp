/**
 * @file   GridEnvironment.hpp
 * @author Nicolas Carion
 * @date   Wed Jun  3 08:26:10 2015
 * 
 * @brief This file implements some simple grid environment, designed for testing purposes
 * 
 * 
 */


#ifndef GRIDENV_H
#define GRIDENV_H

#include "../Environment.hpp"
#include<vector>
#include <stdexcept>

template<typename FeatureComputer>
class GridEnvironment : public t_Environment<FeatureComputer>
{
public:
    typedef typename FeatureComputer::FeatureType FeatureType;
    GridEnvironment(FeatureComputer* feat) : t_Environment<FeatureComputer>(feat),m_width(10),m_height(10),m_frame(0) {reset();}

    void reset(){
        m_posx=0;
        m_posy=0;
    }
    std::vector<Action> getLegalActionSet()
    {
        return {PLAYER_A_UP,PLAYER_A_DOWN,PLAYER_A_LEFT,PLAYER_A_RIGHT};
    }

    void getCompleteFeatureVector(std::vector<FeatureType>& features)
    {
        this->m_feat->getCompleteFeatureVector(this,features);
    }

    void getActiveFeaturesIndices(std::vector<std::pair<int,FeatureType> >& active_feat)
    {
        return this->m_feat->getActiveFeaturesIndices(this,active_feat);
    }

    void getActiveFeaturesIndices(std::vector<int>& active_feat)
    {
        std::vector<std::pair<int,FeatureType> > temp;
        getActiveFeaturesIndices(temp);
        active_feat.clear();
        for(const auto& p : temp){
            active_feat.push_back(p.first);
        }
    }

    int getNumberOfFeatures()
    {
        return this->m_feat->getNumberOfFeatures(this);
    }

    double act(Action action)
    {
        switch(action){
        case PLAYER_A_UP:
            m_posy--;
            break;
        case PLAYER_A_DOWN:
            m_posy++;
            break;
        case PLAYER_A_LEFT:
            m_posx--;
            break;
        case PLAYER_A_RIGHT:
            m_posx++;
            break;
        default:
            throw std::runtime_error("illegal action taken by the agent");
        }
        double reward = -1;
        if(m_posx<0){
            m_posx = 0;
            reward -= 100;
        }
        if(m_posy<0){
            m_posy = 0;
            reward -= 100;
        }
        if(m_posx>m_width){
            m_posx = m_width;
            reward -= 100;
        }
        if(m_posy>m_height){
            m_posy = m_height;
            reward -= 100;
        }
        if(m_posx==m_width&&m_posy==m_height){
            //reward = 10 + m_width + m_height;
            reward = 0;
        }
        m_frame++;
        return reward;
    }

    bool isTerminal(){
        return m_posx==m_width&&m_posy==m_height;
    }

    int getEpisodeFrameNumber(){
        return m_frame;
    }

    std::pair<int,int> getGridSize(){
        return {m_width,m_height};
    }

    std::pair<int,int> getPos(){
        return {m_posx,m_posy};
    }
 protected:
    int m_posx,m_posy;
    int m_width,m_height;
    int m_frame;
};




















#endif
