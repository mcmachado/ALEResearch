/**
 * @file   4_state.hpp
 * @author Marlos C. Machado
 * 
 * @brief This file implements a simple MDP, without any reward.
 *      s1
 *    /    \
 * s0        s3
 *  | \ s2 /  |
 *  |         |
 *   ---------
 */

#ifndef FOURSTATEESNV_H
#define FOURSTATEESNV_H

#include "../Environment.hpp"

template<typename FeatureComputer>
class FourStatesEnvironment : public t_Environment<FeatureComputer>{

protected:
	unsigned int m_frame, m_state;

public:
	typedef typename FeatureComputer::FeatureType FeatureType;
	enum Constants{
		MIN_STATES = 0,
		MAX_STATES = 3,
		NUM_STATES = 4
	};
	FourStatesEnvironment(FeatureComputer* feat) : t_Environment<FeatureComputer>(feat){
		reset();
	}

	virtual void reset() final{
		m_frame = 0;
		/* The MDP has 6 states. One can start in state 1 or 2.*/
		m_state = (rand() % 2)/* + 1*/;
	}
 	
 	std::vector<Action> getMinimalActionSet(){
 		return {PLAYER_A_RIGHT};
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

	void getActiveFeaturesIndices(std::vector<int>& active_feat){
		this->m_feat->getActiveFeaturesIndices(this, active_feat);
    }

    int getNumberOfFeatures(){
    	return NUM_STATES;
    }

    double doAct(Action action){
    	m_frame += 1;

        unsigned int random = rand() % 1000;
        switch(m_state){
            case 0:
                if(random < 500){
                    m_state = 1;
                } else{
                    m_state = 2;
                }
                break;
            case 1:
            case 2:
                m_state = 3;
                break;
            case 3:
                m_state = 0;
                break;
        }
    	return 0.0;
    }

    bool isTerminal(){
    	return false; //This is such a simple MDP that it has no terminal state
    }

    int getEpisodeFrameNumber(){
    	return m_frame;
    }

    int getCurrentState(){
    	return m_state;
    }
};

#endif