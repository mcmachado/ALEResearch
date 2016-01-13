/**
 * @file   RiverSwimEnvironment.hpp
 * @author Marlos C. Machado
 * 
 * @brief This file implements RiverSwim, a domain from Strehl and Littman (2008). 
 * 
 */

#ifndef RIVERSWIMENV_H
#define RIVERSWIMENV_H

#include "../Environment.hpp"

template<typename FeatureComputer>
class RiverSwimEnvironment : public t_Environment<FeatureComputer>{

protected:
	unsigned int m_frame, m_state;

public:
	typedef typename FeatureComputer::FeatureType FeatureType;
	enum Constants{
		MIN_STATES = 0,
		MAX_STATES = 5,
		NUM_STATES = 6 //The regular 6 + bias feature
	};
	RiverSwimEnvironment(FeatureComputer* feat) : t_Environment<FeatureComputer>(feat){
		reset();
	}

	virtual void reset() final{
		m_frame = 0;
		/* The MDP has 6 states. One can start in state 1 or 2.*/
		m_state = (rand() % 2)/* + 1*/;
	}
 	
 	std::vector<Action> getMinimalActionSet(){
 		return {PLAYER_A_LEFT,PLAYER_A_RIGHT};
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
    	return NUM_STATES/* + 1*/;
    }

    double doAct(Action action){
    	m_frame += 1;

    	unsigned int random = rand() % 1000;
    	switch(m_state){
    		case MIN_STATES: //If the agent is in the leftmost state
	    		switch(action){
	    			case PLAYER_A_LEFT:
	    				return 5.0;
	    			case PLAYER_A_RIGHT:
	    				if(random < 300){        //30% go right
	    					m_state = m_state + 1;
	    					return 0.0;
	    				}
	    				else{                    //70% stay where you are
    						return 0.0;
	    				}
	    			default:
	    				throw std::runtime_error("illegal action taken by the agent");
	    		}

    		case MAX_STATES:
	    		switch(action){ //If the agent is in the rightmost state
	    			case PLAYER_A_LEFT:
	    			    m_state = m_state - 1;
    					return 0.0;
	    			case PLAYER_A_RIGHT:
	    				if(random < 300){        //30% stay where you are
	    					return 10000.0;
	    				}
	    				else{                    //70% go left
	    			    	m_state = m_state - 1;
    						return 0.0;
	    				}
	    			default:
	    				throw std::runtime_error("illegal action taken by the agent");
	    		}
    		default:
	    		switch(action){ //If the agent is in any state different from the edges
	    			case PLAYER_A_LEFT: //always go left:
	    			    m_state = m_state - 1;
    					return 0.0;
	    			case PLAYER_A_RIGHT:
		    			if(random < 100){        //10% go left
		    				m_state = m_state - 1;
		    				return 0.0;
		    			} else if(random < 700){ //60% stay where it is
		    				return 0.0;
		    			} else {                 //30% go right
		    				m_state = m_state + 1;
		    				return 0.0;
		    			}
	    			default:
	    				throw std::runtime_error("illegal action taken by the agent");
	    		}
    	}
    }

    bool isTerminal(){
    	//In Strehl and Littman (2008) there is no termination condition,
    	//the task is running for a finite number of steps, that is it.
    	return false;//(m_state == MAX_STATES) || (m_state == MIN_STATES);
    }

    int getEpisodeFrameNumber(){
    	return m_frame;
    }

    int getCurrentState(){
    	return m_state;
    }
};

#endif