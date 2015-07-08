/* Author: Marlos C. Machado */


#ifndef AGENT_H
#define AGENT_H

#include <vector>

class Agent{
	public:
		std::vector<float> freqOfBitFlips; //[0:1023] transitions 0->1; [1024:2048] transitions 1->0
		std::vector<std::vector<std::vector<float> > > w;  //Theta, weights vector
		int numberOfAvailActions, numberOfPrimitiveActions;
	private:
};

#endif