/****************************************************************************************
** Implementation of an agent that is controlled by a human player.
**
** Author: Marlos C. Machado
***************************************************************************************/

#include <iostream>

#include "HumanPlayer.hpp"
#include "SDL.h"

HumanAgent::HumanAgent(){
    #ifndef __USE_SDL
        printf("This code must be compiled with the SDL Library\n");
        exit(1);
    #endif
	maxStepsInEpisode = 18000;
}
		
void HumanAgent::evaluatePolicy(ALEInterface& ale){
#ifdef __USE_SDL
	Action action;
	int reward = 0;

	int step = 0;
	while(!ale.game_over() && step < maxStepsInEpisode) {
		action = receiveAction();
		reward += ale.act(action);
		step++;
	}
	printf("Episode ended with a score of %d points\n", reward);
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
