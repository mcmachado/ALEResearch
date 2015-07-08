/* Author: Marlos C. Machado */

#include <ale_interface.hpp>

#include "../common/Parameters.hpp"

#ifndef AGENT_H
#define AGENT_H

#include <vector>

class Agent{
	public:
		ActionVect actions; //Basic actions
		std::vector<float> freqOfBitFlips; //[0:1023] transitions 0->1; [1024:2048] transitions 1->0
		std::vector<std::vector<std::vector<float> > > w;  //Theta, weights vector
		int numberOfAvailActions, numberOfOptions, numberOfPrimitiveActions;

		Agent(ALEInterface& ale, Parameters *param, long unsigned int numFeatures);
	private:
		Agent();
};

#endif