/****************************************************************************************
** Implementation of an agent that is controlled by a human player.
**
** REMARKS:
**    -- This code is based on the implementation available in the ALE, in the class
**       SDLKeyboardAgent.
**
** Author: Marlos C. Machado
***************************************************************************************/

#include <iostream>

#include "HumanAgent.hpp"
#ifdef __USE_SDL
#include "SDL.h"
#endif

HumanAgent::HumanAgent(Parameters *param){    
    #ifndef __USE_SDL
        printf("This code must be compiled with the SDL Library\n");
        exit(1);
    #endif
	maxStepsInEpisode = param->getEpisodeLength();
	numEpisodesToEval = param->getNumEpisodesEval();
    toSaveTrajectory  = param->getToSaveTrajectory();
    trajectoryFile    = param->getSaveTrajectoryPath();

	// If not displaying the screen, there is little point in having keyboard control
	bool display_screen = param->getDisplay();
	if (!display_screen) {
		printf("Keyboard agent needs DISPLAY = 1.");
		exit(1);
	}
}

#ifdef __USE_SDL

void HumanAgent::saveTrajectory(int takenAction){
    ofstream myfile (this->trajectoryFile, ios::app);
    
    if (myfile.is_open()){
        //Print action taken in the time step:
        myfile << takenAction << ",";
        myfile.close();
    } else {
        printf("I was not able to open the specified file!\n");
        exit(0);
    }    
}
#endif

void HumanAgent::learnPolicy(ALEInterface& ale, Features *features){
}
		
void HumanAgent::evaluatePolicy(ALEInterface& ale, Features *features){
#ifdef __USE_SDL
	Action action;
	int reward = 0;
	int cumulativeReward = 0;
	
	//Repeat (for each episode):
	for(int episode = 0; episode < numEpisodesToEval; episode++){
		int step = 0;
		while(!ale.game_over() && step < maxStepsInEpisode) {
			action = receiveAction();
            //If one wants to save trajectories, this is where the trajectory is saved:
            if(toSaveTrajectory){
                saveTrajectory(action);
            }
			reward = ale.act(action);
			cumulativeReward += reward;
			step++;
		}
		printf("Episode %d, Cumulative Reward: %d\n", episode + 1, cumulativeReward);
		cumulativeReward = 0;
		ale.reset_game(); //Start the game again when the episode is over
	}
}


Action HumanAgent::receiveAction(){

	Action a = PLAYER_A_NOOP;
        // This loop is necessary because keypress events come in quickly
        //SDL_Delay(50); // Set amount of sleep time
        SDL_PumpEvents();
        Uint8* keymap = SDL_GetKeyState(NULL);

            // Triple Actions
        if (keymap[SDLK_UP] && keymap[SDLK_RIGHT] && keymap[SDLK_LSHIFT]) {
            a = PLAYER_A_UPRIGHTFIRE;
        } else if (keymap[SDLK_UP] && keymap[SDLK_LEFT] && keymap[SDLK_LSHIFT]) {
            a = PLAYER_A_UPLEFTFIRE;
        } else if (keymap[SDLK_DOWN] && keymap[SDLK_RIGHT] && keymap[SDLK_LSHIFT]) {
            a = PLAYER_A_DOWNRIGHTFIRE;
        } else if (keymap[SDLK_DOWN] && keymap[SDLK_LEFT] && keymap[SDLK_LSHIFT]) {
            a = PLAYER_A_DOWNLEFTFIRE;

            // Double Actions
        } else if (keymap[SDLK_UP] && keymap[SDLK_LEFT]) {
            a = PLAYER_A_UPLEFT;
        } else if (keymap[SDLK_UP] && keymap[SDLK_RIGHT]) {
            a = PLAYER_A_UPRIGHT;
        } else if (keymap[SDLK_DOWN] && keymap[SDLK_LEFT]) {
            a = PLAYER_A_DOWNLEFT;
        } else if (keymap[SDLK_DOWN] && keymap[SDLK_RIGHT]) {
            a = PLAYER_A_DOWNRIGHT;
        } else if (keymap[SDLK_UP] && keymap[SDLK_LSHIFT]) {
            a = PLAYER_A_UPFIRE;
        } else if (keymap[SDLK_DOWN] && keymap[SDLK_LSHIFT]) {
            a = PLAYER_A_DOWNFIRE;
        } else if (keymap[SDLK_LEFT] && keymap[SDLK_LSHIFT]) {
            a = PLAYER_A_LEFTFIRE;
        } else if (keymap[SDLK_RIGHT] && keymap[SDLK_LSHIFT]) {
            a = PLAYER_A_RIGHTFIRE;

            // Single Actions
        } else if (keymap[SDLK_LSHIFT]) {
            a = PLAYER_A_FIRE;
        } else if (keymap[SDLK_RETURN]) {
            a = PLAYER_A_NOOP;
        } else if (keymap[SDLK_LEFT]) {
            a = PLAYER_A_LEFT;
        } else if (keymap[SDLK_RIGHT]) {
            a = PLAYER_A_RIGHT;
        } else if (keymap[SDLK_UP]) {
            a = PLAYER_A_UP;
        } else if (keymap[SDLK_DOWN]) {
            a = PLAYER_A_DOWN;
        } 
    return a;    
    #endif    
}

HumanAgent::~HumanAgent(){}
