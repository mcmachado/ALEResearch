/**
 * @file   MontainCarEnvironment.hpp
 * @author Nicolas Carion
 * @date   Thu Aug 13 10:32:47 2015
 * 
 * @brief  Implementation of the moutain car environment. The tiling code is borrowed from Rich Sutton.
 * 
 * 
 */


#ifndef MCENV_H
#define MCENV_H

#include "../Environment.hpp"
#include <vector>
#include <stdexcept>
#include <iostream>
#include <cmath>

template<typename FeatureComputer>
class MountainCarEnvironment : public t_Environment<FeatureComputer>{
public:
    typedef typename FeatureComputer::FeatureType FeatureType;
    MountainCarEnvironment(FeatureComputer* feat) :
        t_Environment<FeatureComputer>(feat),
        m_min_pos(-1.2),
        m_max_pos(0.5),
        m_max_abs_vel(0.07),
        m_goal_pos(0.5){
        reset();
        setFlavor(0);
    }

    virtual void reset() final{
        //m_pos = -0.5;
        m_pos = -3.1415926535 / 6.0;
        m_vel = 0.0;
        m_frame = 0;
    }

    std::vector<Action> getLegalActionSet(){
        return {PLAYER_A_NOOP,PLAYER_A_LEFT,PLAYER_A_RIGHT};
    }

    void getCompleteFeatureVector(std::vector<FeatureType>& features){
        this->m_feat->getCompleteFeatureVector(this,features);
    }

    void getActiveFeaturesIndices(std::vector<std::pair<int,FeatureType> >& active_feat){
        return this->m_feat->getActiveFeaturesIndices(this,active_feat);
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
        float accel = 0.0;
        switch(action){
        case PLAYER_A_LEFT:
            accel = -1.0;
            break;
        case PLAYER_A_RIGHT:
            accel = 1.0;
            break;
        case PLAYER_A_NOOP:
            accel = 0;
            break;
        default:
            throw std::runtime_error("illegal action taken by the agent");
        }
        /*m_vel += accel*m_coeff + cos(3.0*m_pos)*(-0.0025);
        m_vel = std::min(m_vel,m_max_abs_vel);
        m_vel = std::max(m_vel,-m_max_abs_vel);
        m_pos += m_vel;
        m_pos = std::min(m_pos,m_max_pos);
        m_pos = std::max(m_pos,m_min_pos);
        if(m_pos == m_min_pos)
            m_vel = std::max(m_vel,float(0.0));
        */
        m_vel = m_vel + accel * m_coeff - 0.0025 * cos(3.0 * m_pos);
        if(m_vel > 0.07){
            m_vel = 0.07;
        }
        if(m_vel < -0.07){
            m_vel = -0.07;
        }
        m_pos = m_pos + m_vel;
        if(m_pos < -1.2){
            m_pos = -1.2;
            m_vel = 0;
        }
        if(m_pos > 0.5){
            m_pos = 0.5;
            m_vel = 0;
        }
        //std::cout<<cos(3*m_pos)<<"vel "<<m_vel<<" pos "<<m_pos<<std::endl;
        
        double reward = -1;
        m_frame++;
        return reward;
    }

    bool isTerminal(){
        return m_pos == m_max_pos || m_frame>10000;
    }

    int getEpisodeFrameNumber(){
        return m_frame;
    }

    float getVel(){
        return m_vel;
    }

    float getPos(){
        return m_pos;
    }

    virtual void setFlavor(unsigned f) override final{
        std::cout<<"flavr "<<f<<std::endl;
        m_coeff = 0.001;
        switch(f){
        case 1:
            m_coeff *= 0.8;
            break;
        case 2:
            m_coeff *= 0.5;
            break;
        case 3:
            m_coeff *= 0.4;
            break;
        case 4:
            m_coeff *= 0.3;
            break;
        case 0:
        default:
            m_coeff=0.001;
            break;
        }
        std::cout<<"coeff "<<m_coeff<<std::endl;
    }
 protected:
    float m_pos, m_vel, m_min_pos, m_max_pos, m_max_abs_vel, m_goal_pos, m_coeff;
    int m_frame;
};


#endif