/**
 * @file   GridEnvironment.hpp
 * @author Nicolas Carion, Marlos
 * 
 * @brief This file implements some simple grid environment. 
 * Mode 1: Just a -1 everywhere until reaching the goal (top right corner).
 * Mode 2: Rewards on opposite corners (top left, bottom right), 
 *           these rewards are stochastic (normally distributed) and they have huge variance.
 * 
 */

#ifndef GRIDENV_H
#define GRIDENV_H

#include "../Environment.hpp"
#include <vector>
#include <random>
#include <iostream>
#include <stdexcept>

template<typename FeatureComputer>
class GridEnvironment : public t_Environment<FeatureComputer>{

private:
    std::default_random_engine generator;
    std::normal_distribution<double> distributionTL;
    std::normal_distribution<double> distributionBR;

public:
    typedef typename FeatureComputer::FeatureType FeatureType;
    GridEnvironment(FeatureComputer* feat, int seed = 1) : 
        t_Environment<FeatureComputer>(feat), distributionTL(3.0, 9.0), distributionBR(0.5, 0.1) {

        generator.seed(seed); 
        m_frame = 0;
        reset();
        setFlavor(0);
    }

    virtual void reset() final{
        m_posx = 0;
        m_posy = 0;
        m_frame = 0;
    }

    std::vector<Action> getMinimalActionSet(){
        return {PLAYER_A_UP, PLAYER_A_DOWN, PLAYER_A_LEFT, PLAYER_A_RIGHT};
    }

    std::vector<Action> getLegalActionSet(){
        return {PLAYER_A_NOOP, PLAYER_A_FIRE, PLAYER_A_UP, PLAYER_A_RIGHT, PLAYER_A_LEFT,
            PLAYER_A_DOWN, PLAYER_A_UPRIGHT, PLAYER_A_UPLEFT, PLAYER_A_DOWNRIGHT, PLAYER_A_DOWNLEFT,
            PLAYER_A_UPFIRE, PLAYER_A_RIGHTFIRE, PLAYER_A_LEFTFIRE, PLAYER_A_DOWNFIRE,
            PLAYER_A_UPRIGHTFIRE, PLAYER_A_UPLEFTFIRE, PLAYER_A_DOWNRIGHTFIRE, PLAYER_A_DOWNLEFTFIRE};
    }

    void getCompleteFeatureVector(std::vector<FeatureType>& features){
        this->m_feat->getCompleteFeatureVector(this,features);
    }

    void getActiveFeaturesIndices(std::vector<std::pair<int,FeatureType> >& active_feat){
        this->m_feat->getActiveFeaturesIndices(this,active_feat);
    }

    void getActiveFeaturesIndices(std::vector<int>& active_feat){
        std::vector<std::pair<int,FeatureType> > temp;
        getActiveFeaturesIndices(temp);
        active_feat.clear();
        for(const auto& p : temp){
            active_feat.push_back(p.first);
        }
    }

    int getNumberOfFeatures(){
        return this->m_feat->getNumberOfFeatures(this);
    }

    double doAct(Action action){
//        printf("Grid: (%d, %d) -> ", m_posx, m_posy);
        switch(action){
        case PLAYER_A_UP:
//            printf("UP -> ");
            m_posy++;
            break;
        case PLAYER_A_DOWN:
//            printf("DOWN -> ");
            m_posy--;
            break;
        case PLAYER_A_LEFT:
//            printf("LEFT -> ");
            m_posx--;
            break;
        case PLAYER_A_RIGHT:
//            printf("RIGHT -> ");
            m_posx++;
            break;
        default:
            throw std::runtime_error("illegal action taken by the agent");
        }

        double reward = m_flavor == 0 ? -1 : 0;
        
        if(m_posx < 0){
            m_posx = 0;
        }
        if(m_posy < 0){
            m_posy = 0;
        }
        if(m_posx > m_width){
            m_posx = m_width;
        }
        if(m_posy > m_height){
            m_posy = m_height;
        }
//        printf("(%d, %d)\n", m_posx, m_posy);
        if(m_flavor == 1 && m_posx == 0 && m_posy == m_height){ 
            reward += distributionTL(generator);
        }

        if(m_flavor == 1 && m_posx == m_width && m_posy == 0){
            reward += distributionBR(generator);
        }

        m_frame++;
        return reward;
    }

    bool isTerminal(){
        bool reachedTopLeftCorner = m_posx == 0 && m_posy == m_height;
        bool reachedBottomRightCorner = m_posx == m_width && m_posy == 0;

        switch(m_flavor){
            case 1:
                if(reachedTopLeftCorner){
//                    printf("Terminal: (0, 9)\n");
                }
                if(reachedBottomRightCorner){
//                    printf("Terminal: (9, 0)\n");
                }
                return reachedTopLeftCorner || reachedBottomRightCorner;
            case 0:
            default:
                return m_posx == m_width && m_posy == m_height;
        }
    }

    int getEpisodeFrameNumber(){
        return m_frame;
    }

    std::pair<int,int> getGridSize(){
        return {m_width, m_height};
    }

    std::pair<int,int> getPos(){
        return {m_posx, m_posy};
    }

    virtual void setFlavor(unsigned f) override final{
        
        //std::cout << "flavor " << f << std::endl;

        switch(f){
        case 1:
            m_width = 10;
            m_height = 10;
            m_flavor = 1;
            break;
        case 0:
            m_width = 10;
            m_height = 10;
            m_flavor = 0;
        default:
            break;
        }
    }

 protected:
    int m_posx, m_posy;
    int m_width, m_height;
    int m_frame, m_flavor;
};

#endif
